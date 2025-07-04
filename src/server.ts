import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import multer from 'multer';
import * as tf from '@tensorflow/tfjs-node';
import { ModelService } from './services/ModelService';
import { ImageProcessor } from './services/ImageProcessor';
import { ClassificationController } from './controllers/ClassificationController';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// File upload configuration
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req: any, file: any, cb: any) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'));
    }
  },
});

// Initialize services
const modelService = new ModelService('./model_js');
const imageProcessor = new ImageProcessor();
const classificationController = new ClassificationController(modelService, imageProcessor);

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'TensorFlow Image Classification API',
    version: '1.0.0',
    endpoints: {
      'POST /classify': 'Upload an image for classification',
      'GET /health': 'Health check endpoint',
    },
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    modelLoaded: modelService.isModelLoaded(),
  });
});

app.post('/classify', upload.single('image') as any, classificationController.classifyImage.bind(classificationController));

app.get('/model-info', classificationController.getModelInfo.bind(classificationController));

// Error handling middleware
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Error:', error);

  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      error: 'File upload error',
      message: error.message,
    });
  }

  res.status(500).json({
    error: 'Internal server error',
    message: error.message || 'Something went wrong',
  });
  return;
});

// Start server
async function startServer() {
  try {
    console.log('Loading TensorFlow model...');
    await modelService.loadModel();
    console.log('Model loaded successfully');

    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
      console.log(`Health check: http://localhost:${PORT}/health`);
      console.log(`Classification endpoint: http://localhost:${PORT}/classify`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();
