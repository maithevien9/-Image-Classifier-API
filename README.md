convert model: tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./model ./model_js

# TensorFlow Image Classification API

A Node.js Express server with TypeScript that serves a CIFAR-10 image classification model using TensorFlow.js.

## Features

- 🚀 Express.js server with TypeScript
- 🧠 TensorFlow.js model loading and inference
- 📸 Image preprocessing and validation
- 🔍 CIFAR-10 classification (10 classes)
- 📊 Detailed prediction results with confidence scores
- 🛡️ Input validation and error handling
- 📈 Health monitoring endpoints

## CIFAR-10 Classes

The model can classify images into these 10 categories:

- ✈️ plane
- 🚗 car
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐕 dog
- 🐸 frog
- 🐴 horse
- 🚢 ship
- 🚛 truck

## Setup

### Prerequisites

- Node.js 18+
- Yarn package manager
- A trained TensorFlow SavedModel in the `./model` directory

### Installation

1. Install dependencies:

```bash
yarn install
```

2. Build the TypeScript code:

```bash
yarn build
```

3. Start the development server:

```bash
yarn dev
```

Or start the production server:

```bash
yarn start
```

The server will start on `http://localhost:3000` by default.

## API Endpoints

### Health Check

```http
GET /health
```

Returns server status and model loading state.

### Root Info

```http
GET /
```

Returns API information and available endpoints.

### Image Classification

```http
POST /classify
```

Upload an image for classification.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `image` field containing the image file

**Response:**

```json
{
  "success": true,
  "result": {
    "predictedClass": "cat",
    "confidence": 0.8542,
    "confidencePercentage": "85.42",
    "allPredictions": [
      {
        "class": "cat",
        "probability": 0.8542,
        "percentage": "85.42"
      },
      {
        "class": "dog",
        "probability": 0.1234,
        "percentage": "12.34"
      }
    ]
  },
  "metadata": {
    "originalImage": {
      "filename": "cat.jpg",
      "size": 245760,
      "mimetype": "image/jpeg",
      "dimensions": "500x400"
    },
    "processing": {
      "timeMs": 156,
      "modelInputSize": "32x32x3",
      "normalization": "Range [-1, 1]"
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:3000/health

# Classify an image
curl -X POST \
  -F "image=@path/to/your/image.jpg" \
  http://localhost:3000/classify
```

### Using JavaScript fetch

```javascript
const formData = new FormData();
formData.append('image', imageFile);

const response = await fetch('http://localhost:3000/classify', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log('Prediction:', result.result.predictedClass);
```

### Using Python requests

```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:3000/classify', files=files)
    result = response.json()
    print(f"Predicted class: {result['result']['predictedClass']}")
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- GIF (.gif)
- BMP (.bmp)

## Image Processing

The API automatically:

1. Resizes images to 32x32 pixels (CIFAR-10 input size)
2. Normalizes pixel values to range [-1, 1]
3. Converts to RGB format if needed
4. Adds batch dimension for model inference

## Error Handling

The API provides detailed error messages for:

- Missing or invalid image files
- Unsupported image formats
- Model loading failures
- Processing errors

Example error response:

```json
{
  "error": "Invalid image",
  "message": "Unsupported image format: bmp. Supported: jpeg, jpg, png, webp, gif",
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## Development

### Project Structure

```
src/
├── controllers/
│   └── ClassificationController.ts
├── services/
│   ├── ModelService.ts
│   └── ImageProcessor.ts
└── server.ts
```

### Scripts

- `yarn dev` - Start development server with hot reload
- `yarn build` - Compile TypeScript to JavaScript
- `yarn start` - Start production server
- `yarn type-check` - Check TypeScript types

## Model Requirements

Your TensorFlow SavedModel should be located in the `./model` directory with this structure:

```
model/
├── saved_model.pb
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── assets/ (optional)
```

The model should:

- Accept input shape `[batch_size, 32, 32, 3]`
- Expect normalized pixel values in range `[-1, 1]`
- Output 10 class probabilities for CIFAR-10 categories

## Performance Tips

- Use production mode (`NODE_ENV=production`) for better performance
- Consider using a reverse proxy (nginx) for production deployments
- Monitor memory usage as TensorFlow.js can be memory-intensive
- Implement rate limiting for production use

## License

MIT License
