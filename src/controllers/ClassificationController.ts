import { Request, Response } from 'express';
import { ModelService } from '../services/ModelService';
import { ImageProcessor } from '../services/ImageProcessor';

export class ClassificationController {
  constructor(private readonly modelService: ModelService, private readonly imageProcessor: ImageProcessor) {}

  async classifyImage(req: Request, res: Response, next: unknown): Promise<void> {
    try {
      // Validate file upload
      if (!req.file) {
        res.status(400).json({
          error: 'No image file provided',
          message: 'Please upload an image file using the "image" field',
        });
        return;
      }

      // Validate model is loaded
      if (!this.modelService.isModelLoaded()) {
        res.status(503).json({
          error: 'Model not available',
          message: 'The classification model is not loaded yet. Please try again later.',
        });
        return;
      }

      const startTime = Date.now();

      // Validate and get image metadata
      const imageMetadata = await this.imageProcessor.validateImage(req.file.buffer);
      console.log('Processing image:', {
        originalName: req.file.originalname,
        size: req.file.size,
        mimetype: req.file.mimetype,
        dimensions: `${imageMetadata.width}x${imageMetadata.height}`,
      });

      // Preprocess image for model
      const preprocessedImage = await this.imageProcessor.preprocessImage(req.file.buffer);

      // Make prediction
      const prediction = await this.modelService.predict(preprocessedImage);

      // Cleanup tensor
      preprocessedImage.dispose();

      const processingTime = Date.now() - startTime;

      // Return successful response
      res.json({
        success: true,
        result: {
          predictedClass: prediction.predictedClass,
          confidence: prediction.confidence,
          confidencePercentage: prediction.confidencePercentage,
          allPredictions: prediction.allPredictions.slice(0, 5), // Top 5 predictions
        },
        metadata: {
          originalImage: {
            filename: req.file.originalname,
            size: req.file.size,
            mimetype: req.file.mimetype,
            dimensions: `${imageMetadata.width}x${imageMetadata.height}`,
          },
          processing: {
            timeMs: processingTime,
            modelInputSize: '32x32x3',
            normalization: 'Range [-1, 1]',
          },
        },
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error('Classification error:', error);

      // Handle specific error types
      if (error instanceof Error) {
        if (error.message.includes('Image validation failed') || error.message.includes('Image preprocessing failed')) {
          res.status(400).json({
            error: 'Invalid image',
            message: error.message,
            timestamp: new Date().toISOString(),
          });
          return;
        }

        if (error.message.includes('Model not loaded')) {
          res.status(503).json({
            error: 'Service unavailable',
            message: 'Classification model is not ready',
            timestamp: new Date().toISOString(),
          });
          return;
        }
      }

      // Generic error response
      res.status(500).json({
        error: 'Classification failed',
        message: 'An error occurred during image classification',
        timestamp: new Date().toISOString(),
      });
    }
  }

  async getModelInfo(req: Request, res: Response): Promise<void> {
    try {
      res.json({
        modelLoaded: this.modelService.isModelLoaded(),
        classes: this.modelService.getClasses(),
        modelInfo: {
          type: 'CIFAR-10 Image Classifier',
          inputShape: [32, 32, 3],
          outputClasses: 10,
          normalization: 'Range [-1, 1]',
          supportedFormats: ['jpeg', 'jpg', 'png', 'webp', 'gif', 'bmp'],
        },
      });
    } catch (error) {
      console.error('Error getting model info:', error);
      res.status(500).json({
        error: 'Failed to get model information',
        message: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
}
