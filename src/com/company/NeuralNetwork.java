package com.company;
import java.io.*;
import java.util.Arrays;
import java.util.Collections;


public class NeuralNetwork
{
    // neuron arrays
    private final Neuron[] inputLayer;
    private final Neuron[][] midLayers;
    private final Neuron[] outputLayer;

    // constructor takes all size constraints of network, creates neuron arrays
    // based on constraints, and initializes the neurons in the arrays
    public NeuralNetwork(int numInputs, int numMidLayers, int numMidLayerNodes, int numOutputLayerNodes)
    {
        // initialize neuron arrays
        inputLayer = new Neuron[numInputs];
        midLayers = new Neuron[numMidLayers][numMidLayerNodes];
        outputLayer = new Neuron[numOutputLayerNodes];

        // initialize neurons in input layer
        for (int i = 0; i < numInputs; i++)
            inputLayer[i] = new Neuron();

        // initialize neurons in first mid layer
        for (int i = 0; i < numMidLayerNodes; i++)
            midLayers[0][i] = new Neuron(inputLayer);

        // initialize neurons in all other mid layers
        for (int i = 1; i < numMidLayers; i++)
        {
            for (int j = 0; j < numMidLayerNodes; j++)
                midLayers[i][j] = new Neuron(midLayers[i-1]);
        }

        // initialize neurons in final layer
        for (int i = 0; i < numOutputLayerNodes; i++)
            outputLayer[i] = new Neuron(midLayers[numMidLayers-1]);
    }

    // loads network from text file at provided file path
    // If network parameters at beginning of file do not match network
    // parameters are trying to be loaded onto, halts loading operation and prints error message
    public boolean LoadNetwork(String filePath)
    {
        // get network parameters
        int[] networkParams = new int[]{inputLayer.length, midLayers.length, midLayers[0].length, outputLayer.length};

        try
        {
            // create BufferedReader based on given file path
            File networkFile = new File(filePath);
            BufferedReader fileReader = new BufferedReader(new FileReader(networkFile));

            // check network parameters
            for (int param : networkParams)
            {
                if (param != Integer.parseInt(fileReader.readLine()))
                {
                    System.out.println("Given network file does not match network parameters, halting load\n");
                    return false;
                }
            }

            // TODO: Make file input parser for network loading

            // close BufferedReader
            fileReader.close();

            return true;
        }
        catch (FileNotFoundException e)
        {
            System.out.println("The specified file could not be found: \"" + filePath + "\"");
            return false;
        }
        catch (IOException e)
        {
            System.out.println("An error occurred while reading the training data file");
            return false;
        }
    }

    // manually set weights and biases of network based on parameters
    public void LoadNetwork(double[][] midLayerBiases, double[][][] midLayerWeights, double[] outputLayerBiases, double[][] outputLayerWeights)
    {
        // set mid layer biases and weights
        for (int i = 0; i < midLayers.length; i++)
        {
            for (int j = 0; j < midLayers[i].length; j++)
            {
                midLayers[i][j].SetBias(midLayerBiases[i][j]);
                midLayers[i][j].SetWeights(midLayerWeights[i][j]);
            }
        }

        // set final layer biases and weights
        for (int i = 0; i < outputLayer.length; i++)
        {
            outputLayer[i].SetBias(outputLayerBiases[i]);
            outputLayer[i].SetWeights(outputLayerWeights[i]);
        }
    }

    // output weights and biases of network to text file for later loading
    /* Output Format
        numInputs
        numMidLayers
        numMidLayerNodes
        numOutputLayerNodes

    */
    public void SaveNetwork()
    {
        // TODO: Implement Save Network
    }

    // runs training algorithms on networks using the given training set,
    // learning rate, mini-batch size, and number of epochs
    public void TrainNetwork(NetworkInput[] trainingData, double learnRate, int batchSize, int numEpochs)
    {
        // loop through set number of epochs
        for (int i = 0; i < numEpochs; i++)
        {
            if (!Main.DEBUG)
            {
                // shuffle training data
                Collections.shuffle(Arrays.asList(trainingData));
            }

            // array stores a mini-batch during use
            NetworkInput[] miniBatch = new NetworkInput[batchSize];
            
            // loop through mini-batches
            for (int j = 0; j < trainingData.length; j += batchSize)
            {
                // copy a subset of the training data into miniBatch
                System.arraycopy(trainingData, j, miniBatch, 0, batchSize);

                // run mini-batch
                RunMiniBatch(miniBatch, learnRate);
            }

            System.out.println("Finished epoch " + (i + 1));
        }
    }

    // runs mini-batch, running back propagation algorithm on each input in batch
    private void RunMiniBatch(NetworkInput[] miniBatch, double learnRate)
    {
        // run and calculate gradient for each case in mini batch
        for (NetworkInput networkInput : miniBatch)
        {
            RunNetwork(networkInput.inputValues);
            BackPropagate(networkInput.correctOutputVector);
        }

        // apply gradients to all non-input neurons
        // apply gradients for mid layers
        for (Neuron[] layer : midLayers)
        {
            for (Neuron neuron : layer)
                neuron.ApplyGradients(learnRate, miniBatch.length);
        }

        // apply gradients for final layer
        for (Neuron neuron : outputLayer)
            neuron.ApplyGradients(learnRate, miniBatch.length);
    }

    // runs back propagation algorithm for current training case
    private void BackPropagate(int[] correctOutputVector)
    {
        // compute bias gradient for output layer
        for (int i = 0; i < outputLayer.length; i++)
        {
            double biasGradient = (outputLayer[i].GetActivation() - correctOutputVector[i]) * outputLayer[i].GetActivation() * (1 - outputLayer[i].GetActivation());
            outputLayer[i].SetBiasGradient(biasGradient);
        }
        // compute weight gradient for output layer
        for (Neuron neuron : outputLayer)
        {
            double[] weightGradient = new double[neuron.GetInputs().length];
            for (int j = 0; j < neuron.GetInputs().length; j++)
                weightGradient[j] = neuron.GetInputs()[j].GetActivation() * neuron.GetBiasGradient();

            neuron.SetWeightGradient(weightGradient);
        }

        // compute bias and weight gradients for each layer moving backwards
        for (int i = midLayers.length - 1; i >= 0; i--)
        {
            // compute bias gradient for current layer
            for (int j = 0; j < midLayers[i].length; j++)
            {
                double sumOfBiasScaledWeights = 0;
                if (i + 1 == midLayers.length)  // if on final hidden layer
                {
                    // calculate Sum(Weight[jk] * BiasGradient[j])
                    for (Neuron neuron : outputLayer)
                        sumOfBiasScaledWeights += neuron.GetWeights()[j] * neuron.GetBiasGradient();

                    // calculate and set biasGradient
                    double biasGradient = sumOfBiasScaledWeights * midLayers[i][j].GetActivation() * (1 - midLayers[i][j].GetActivation());
                    midLayers[i][j].SetBiasGradient(biasGradient);
                }
                else  // for all other hidden layers
                {
                    // calculate Sum(Weight[jk] * BiasGradient[j])
                    for (Neuron neuron : midLayers[i + 1])
                        sumOfBiasScaledWeights += neuron.GetWeights()[j] * neuron.GetBiasGradient();

                    // calculate and set biasGradient
                    double biasGradient = sumOfBiasScaledWeights * midLayers[i][j].GetActivation() * (1 - midLayers[i][j].GetActivation());
                    midLayers[i][j].SetBiasGradient(biasGradient);
                }
            }

            // compute weight gradients for current layer
            for (Neuron neuron : midLayers[i])
            {
                double[] weightGradient = new double[neuron.GetInputs().length];
                for (int j = 0; j < neuron.GetInputs().length; j++)
                    weightGradient[j] = neuron.GetInputs()[j].GetActivation() * neuron.GetBiasGradient();

                neuron.SetWeightGradient(weightGradient);
            }
        }
    }

    // runs through test data set and gives back array of correctly answered inputs
    public boolean[] TestNetwork(NetworkInput[] testingData)
    {
        // track which cases were answered correctly and incorrectly
        boolean[] correctlyAnsweredInputs = new boolean[testingData.length];

        // iterate through testing data, marking correct and incorrect cases
        for (int i = 0; i < testingData.length; i++)
            correctlyAnsweredInputs[i] = (testingData[i].correctOutput == RunNetwork(testingData[i].inputValues));

        return correctlyAnsweredInputs;
    }

    // sets the output of each of the input neurons as the input data
    // from the provided NetworkInput object
    private void InputNetworkData(double[] input)
    {
        for (int i = 0; i < inputLayer.length; i++)
            inputLayer[i].SetActivation(input[i]);
    }

    // run network on given NetworkInput, can be accessed from outside class to run on arbitrary data
    // inputs must be normalized between 0.0-1.0
    public int RunNetwork(double[] input)
    {
        // set input neurons to given input
        InputNetworkData(input);

        // calculate activation in middle layers
        for (Neuron[] layer : midLayers)
        {
            for (Neuron neuron : layer)
                neuron.CalculateActivation();
        }

        // calculate activation for output layer
        for (Neuron neuron : outputLayer)
            neuron.CalculateActivation();

        // get max activation in output layer
        int maxActivation = 0;  // current highest activation index
        for (int i = 0; i < outputLayer.length; i++)
        {
            // if the activation of the i-th neuron is greater than previous max, set new max
            if (outputLayer[i].GetActivation() > outputLayer[maxActivation].GetActivation())
                maxActivation = i;
        }

        return maxActivation;
    }
}
