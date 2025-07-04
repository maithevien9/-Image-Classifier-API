import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';

export class ImageProcessor {
  private readonly TARGET_WIDTH = 32;
  private readonly TARGET_HEIGHT = 32;

  /**
   * Preprocess image for CIFAR-10 model:
   * 1. Resize to 32x32 pixels
   * 2. Normalize pixel values to range [-1, 1]
   * 3. Convert to tensor with shape [1, 32, 32, 3]
   */
  async preprocessImage(imageBuffer: Buffer): Promise<tf.Tensor> {
    try {
      // Resize image to 32x32 and ensure RGB format
      const processedBuffer = await sharp(imageBuffer)
        .resize(this.TARGET_WIDTH, this.TARGET_HEIGHT, {
          fit: 'fill', // Stretch to exact dimensions
          background: { r: 0, g: 0, b: 0 },
        })
        .png()
        .toBuffer();

      // Convert buffer to tensor
      const imageTensor = tf.node.decodeImage(processedBuffer, 3) as tf.Tensor3D;

      // Ensure the image has the right shape
      const resizedTensor = tf.image.resizeBilinear(imageTensor, [this.TARGET_WIDTH, this.TARGET_HEIGHT]);

      // Normalize pixel values: (pixel_value / 127.5) - 1 to get range [-1, 1]
      const normalizedTensor = tf.div(tf.sub(resizedTensor, 127.5), 127.5);

      // Add batch dimension: [32, 32, 3] -> [1, 32, 32, 3]
      const batchedTensor = tf.expandDims(normalizedTensor, 0);

      // Cleanup intermediate tensors
      imageTensor.dispose();
      resizedTensor.dispose();
      normalizedTensor.dispose();

      return batchedTensor;
    } catch (error) {
      console.error('Error preprocessing image:', error);
      throw new Error(`Image preprocessing failed: ${error}`);
    }
  }

  /**
   * Validate image format and size
   */
  async validateImage(imageBuffer: Buffer): Promise<ImageMetadata> {
    try {
      const metadata = await sharp(imageBuffer).metadata();

      if (!metadata.width || !metadata.height) {
        throw new Error('Unable to determine image dimensions');
      }

      const supportedFormats = ['jpeg', 'jpg', 'png', 'webp', 'gif', 'bmp'];
      if (!metadata.format || !supportedFormats.includes(metadata.format)) {
        throw new Error(`Unsupported image format: ${metadata.format}. Supported: ${supportedFormats.join(', ')}`);
      }

      return {
        width: metadata.width,
        height: metadata.height,
        format: metadata.format,
        size: imageBuffer.length,
        channels: metadata.channels || 3,
      };
    } catch (error) {
      console.error('Error validating image:', error);
      throw new Error(`Image validation failed: ${error}`);
    }
  }

  /**
   * Create a debug version of preprocessed image for visualization
   */
  async createDebugImage(preprocessedTensor: tf.Tensor): Promise<Buffer> {
    try {
      // Remove batch dimension and denormalize
      const squeezed = tf.squeeze(preprocessedTensor) as tf.Tensor3D;
      const denormalized = tf.add(tf.mul(squeezed, 127.5), 127.5);
      const clamped = tf.clipByValue(denormalized, 0, 255);
      const uint8Tensor = tf.cast(clamped, 'int32') as tf.Tensor3D;

      // Convert to buffer
      const encodedImage = await tf.node.encodeJpeg(uint8Tensor, 'rgb', 90);
      const debugBuffer = Buffer.from(encodedImage);

      // Cleanup tensors
      squeezed.dispose();
      denormalized.dispose();
      clamped.dispose();
      uint8Tensor.dispose();

      return debugBuffer;
    } catch (error) {
      console.error('Error creating debug image:', error);
      throw new Error(`Debug image creation failed: ${error}`);
    }
  }
}

export interface ImageMetadata {
  width: number;
  height: number;
  format: string;
  size: number;
  channels: number;
}
