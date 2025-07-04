import * as tf from '@tensorflow/tfjs-node';

export class ModelService {
  private model: tf.GraphModel | null = null;
  private readonly modelPath: string;
  private readonly classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

  constructor(modelPath: string) {
    this.modelPath = modelPath;
  }

  async loadModel(): Promise<void> {
    try {
      console.log(`Loading model from: ${this.modelPath}`);
      this.model = await tf.loadGraphModel(`file://${this.modelPath}/model.json`);
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      throw new Error(`Failed to load model: ${error}`);
    }
  }

  async predict(preprocessedImage: tf.Tensor): Promise<PredictionResult> {
    if (!this.model) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    try {
      // Make prediction
      const prediction = this.model.predict(preprocessedImage) as tf.Tensor;
      const probabilities = await prediction.data();

      // Get the class with highest probability
      const maxProbabilityIndex = Array.from(probabilities).indexOf(Math.max(...probabilities));
      const predictedClass = this.classes[maxProbabilityIndex];
      const confidence = probabilities[maxProbabilityIndex];

      // Create detailed results with all class probabilities
      const classResults: ClassProbability[] = this.classes.map((className, index) => ({
        class: className,
        probability: probabilities[index],
        percentage: (probabilities[index] * 100).toFixed(2),
      }));

      // Sort by probability (highest first)
      classResults.sort((a, b) => b.probability - a.probability);

      // Cleanup tensors
      prediction.dispose();

      return {
        predictedClass,
        confidence: Number(confidence.toFixed(4)),
        confidencePercentage: (confidence * 100).toFixed(2),
        allPredictions: classResults,
      };
    } catch (error) {
      console.error('Error during prediction:', error);
      throw new Error(`Prediction failed: ${error}`);
    }
  }

  isModelLoaded(): boolean {
    return this.model !== null;
  }

  getClasses(): string[] {
    return [...this.classes];
  }
}

export interface PredictionResult {
  predictedClass: string;
  confidence: number;
  confidencePercentage: string;
  allPredictions: ClassProbability[];
}

export interface ClassProbability {
  class: string;
  probability: number;
  percentage: string;
}
