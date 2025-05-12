package com.jahid.objectd

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner

import com.jahid.objectd.databinding.ActivityCameraBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.util.concurrent.Executors
import kotlin.math.min
import kotlin.random.Random


class CameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCameraBinding
    private lateinit var bitmapBuffer: Bitmap

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private val tfImageBuffer = TensorImage(DataType.UINT8)

    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(NnApiDelegate())
        )
    }

    private val detector by lazy {
        ObjectDetectionHelper(tflite, FileUtil.loadLabels(this, LABELS_PATH))
    }

    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    private val tfImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR
                )
            )
            .add(Rot90Op(imageRotationDegrees / 90))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Listener for button used to capture photo
        binding.cameraCaptureButton.setOnClickListener {
            it.isEnabled = false

            if (pauseAnalysis) {
                pauseAnalysis = false
                binding.imagePredicted.visibility = View.GONE
            } else {
                pauseAnalysis = true
                val matrix = Matrix().apply {
                    postRotate(imageRotationDegrees.toFloat())
                    if (isFrontFacing) postScale(-1f, 1f)
                }
                val uprightImage = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
                )
                binding.imagePredicted.setImageBitmap(uprightImage)
                binding.imagePredicted.visibility = View.VISIBLE
            }

            it.isEnabled = true
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() = binding.viewFinder.post {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(binding.viewFinder.display.rotation)
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(binding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()
            val converter = YuvToRgbConverter(this)

            imageAnalysis.setAnalyzer(executor) { imageProxy ->
                if (!::bitmapBuffer.isInitialized) {
                    imageRotationDegrees = imageProxy.imageInfo.rotationDegrees
                    bitmapBuffer = Bitmap.createBitmap(
                        imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
                    )
                }

                if (pauseAnalysis) {
                    imageProxy.close()
                    return@setAnalyzer
                }

                imageProxy.image?.let { image ->
                    converter.yuvToRgb(image, bitmapBuffer)
                }

                val tfImage = tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
                val predictions = detector.predict(tfImage)

                reportPrediction(predictions.maxByOrNull { it.score })

                val frameCount = 10
                if (++frameCounter % frameCount == 0) {
                    frameCounter = 0
                    val now = System.currentTimeMillis()
                    val delta = now - lastFpsTimestamp
                    val fps = 1000 * frameCount.toFloat() / delta
                    Log.d(TAG, "FPS: ${"%.02f".format(fps)}")
                    lastFpsTimestamp = now
                }

                imageProxy.close()
            }

            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            cameraProvider.unbindAll()
            val camera = cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis
            )

            preview.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun reportPrediction(prediction: ObjectDetectionHelper.ObjectPrediction?) =
        binding.viewFinder.post {
            if (prediction == null || prediction.score < ACCURACY_THRESHOLD) {
                binding.boxPrediction.visibility = View.GONE
                binding.textPrediction.visibility = View.GONE
                return@post
            }

            val location = mapOutputCoordinates(prediction.location)

            binding.textPrediction.text = "${"%.2f".format(prediction.score)} ${prediction.label}"
            (binding.boxPrediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
                topMargin = location.top.toInt()
                leftMargin = location.left.toInt()
                width =
                    min(binding.viewFinder.width, location.right.toInt() - location.left.toInt())
                height =
                    min(binding.viewFinder.height, location.bottom.toInt() - location.top.toInt())
            }

            binding.boxPrediction.visibility = View.VISIBLE
            binding.textPrediction.visibility = View.VISIBLE
        }

    private fun mapOutputCoordinates(location: RectF): RectF {
        val previewLocation = RectF(
            location.left * binding.viewFinder.width,
            location.top * binding.viewFinder.height,
            location.right * binding.viewFinder.width,
            location.bottom * binding.viewFinder.height
        )

        val isFrontFacing = lensFacing == CameraSelector.LENS_FACING_FRONT
        val isFlippedOrientation = imageRotationDegrees == 90 || imageRotationDegrees == 270
        val rotatedLocation = if (
            (!isFrontFacing && isFlippedOrientation) ||
            (isFrontFacing && !isFlippedOrientation)
        ) {
            RectF(
                binding.viewFinder.width - previewLocation.right,
                binding.viewFinder.height - previewLocation.bottom,
                binding.viewFinder.width - previewLocation.left,
                binding.viewFinder.height - previewLocation.top
            )
        } else {
            previewLocation
        }

        val margin = 0.1f
        val requestedRatio = 4f / 3f
        val midX = (rotatedLocation.left + rotatedLocation.right) / 2f
        val midY = (rotatedLocation.top + rotatedLocation.bottom) / 2f
        return if (binding.viewFinder.width < binding.viewFinder.height) {
            RectF(
                midX - (1f + margin) * requestedRatio * rotatedLocation.width() / 2f,
                midY - (1f - margin) * rotatedLocation.height() / 2f,
                midX + (1f + margin) * requestedRatio * rotatedLocation.width() / 2f,
                midY + (1f - margin) * rotatedLocation.height() / 2f
            )
        } else {
            RectF(
                midX - (1f - margin) * rotatedLocation.width() / 2f,
                midY - (1f + margin) * requestedRatio * rotatedLocation.height() / 2f,
                midX + (1f - margin) * rotatedLocation.width() / 2f,
                midY + (1f + margin) * requestedRatio * rotatedLocation.height() / 2f
            )
        }
    }

    override fun onResume() {
        super.onResume()
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode
            )
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        tflite.close()
    }

    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private val TAG = CameraActivity::class.java.simpleName
        private const val ACCURACY_THRESHOLD = 0.5f
        private const val MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
        private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
    }
}