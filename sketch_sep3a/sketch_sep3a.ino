// Layer 1
float weights1[6][64] = {
  {-0.35525432, -0.06947343,  0.1148968, -0.4246893, 0.42569485, -0.1809677, /*...*/ }, 
  {0.37101153, 0.24076286, -0.34818184, -0.32541, 0.4016915, -0.42767495, /*...*/ }, 
  {-0.3546754, -0.5463783, -0.19135687, -0.2812656, -0.460873, -0.4556425, /*...*/ }, 
  {0.3876935, 0.05392601, 0.34767836, -0.30266017, -0.27434084, -0.1217028, /*...*/ }, 
  {-0.41210043, -0.12660336, 0.05148305, 0.32616085, -0.13877073, -0.02175247, /*...*/ }, 
  {-0.28138712, 0.05198949, 0.19750132, 0.04697943, -0.43933517, 0.2919234, /*...*/ }
};
float biases1[64] = {0.2270111, 0.22433521, 0.20658386, 0.19452123, 0.30107313, 0.18781592, /*...*/ };

// Layer 2
float weights2[64][64] = {
  {-0.20436276, 0.22151831, 0.09681877, /*...*/ }, 
  {0.00551542, 0.04124433, -0.21339588, /*...*/ }, 
  {-0.07989531, -0.04689885, -0.2178645, /*...*/ }, 
};
float biases2[64] = {-0.03561414, 0.15843247, -0.05016774, 0.18295668, /*...*/ };

// Layer 3
float weights3[64][1] = {
  {-0.1671649}, {0.47676617}, {-0.2661694}, {0.2758783}, {-0.03668024}, {-0.21434814}, /*...*/ 
};
float biases3[1] = {0.13079621};

// Sigmoid activation function
float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

// Tanh activation function
float tanhActivation(float x) {
  return tanh(x);
}

// Matrix multiplication for non-square matrices
void matmul(const float A[][64], const float B[], float result[], int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    result[i] = 0;
    for (int j = 0; j < cols; j++) {
      result[i] += A[i][j] * B[j];
    }
  }
}

// Matrix multiplication for the output layer (64x1 matrix)
float matmul_output_layer(const float A[], const float B[][1], int size) {
  float result = 0;
  for (int i = 0; i < size; i++) {
    result += A[i] * B[i][0];
  }
  return result;
}

// Normalization/Standardization constants (calculated from training data)
float means[6] = {2013.0, 17.7, 1083.88, 4.09, 24.96, 121.54};
float std_devs[6] = {0.58, 10.14, 1262.11, 2.98, 0.01, 0.02};

// Function to standardize the input
void standardize_input(float input[], float mean[], float std_dev[], int size) {
  for (int i = 0; i < size; i++) {
    input[i] = (input[i] - mean[i]) / std_dev[i];
  }
}

// Neural network forward pass
float forwardPass(float input[]) {
  float layer1_output[64];
  float layer2_output[64];
  float final_output;

  // Layer 1 computation
  matmul(weights1, input, layer1_output, 64, 6);
  for (int i = 0; i < 64; i++) {
    layer1_output[i] = sigmoid(layer1_output[i] + biases1[i]);
  }

  // Layer 2 computation
  matmul(weights2, layer1_output, layer2_output, 64, 64);
  for (int i = 0; i < 64; i++) {
    layer2_output[i] = tanhActivation(layer2_output[i] + biases2[i]);
  }

  // Output Layer computation
  final_output = matmul_output_layer(layer2_output, weights3, 64);
  final_output += biases3[0];

  return final_output;
}

void setup() {
  Serial.begin(9600);  
  float input[6] = {2014.000, 5, 300, 7, 24.980, 121.540};

  // Standardize the input
  standardize_input(input, means, std_devs, 6);

  // Perform a forward pass through the network
  float prediction = forwardPass(input);

  Serial.print("Prediction: ");
  Serial.println(prediction);
}

void loop() {
}
