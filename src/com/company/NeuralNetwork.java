package com.company;

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

    // runs training algorithms on networks using the given training set
    public void TrainNetwork(NetworkInput[] trainingData)
    {
        // TODO: Implement TrainNetwork
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
