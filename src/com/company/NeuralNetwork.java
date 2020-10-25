package com.company;

public class NeuralNetwork
{
    // neuron arrays
    private Neuron[] inputLayer;
    private Neuron[][] midLayers;
    private Neuron[] outputLayer;

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

    public void InputNetworkData(NetworkInput input)
    {
        for (int i = 0; i < inputLayer.length; i++)
            inputLayer[i].SetOutput(input.inputValues[i]);
    }
}
