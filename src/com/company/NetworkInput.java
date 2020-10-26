package com.company;

public class NetworkInput
{
    public int correctOutput;
    public int[] correctOutputVector;
    public double[] inputValues;

    public NetworkInput(int correctOutput, String[] inputValues)
    {
        // initialize variables
        this.correctOutput = correctOutput;
        this.correctOutputVector = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        this.inputValues = new double[inputValues.length];

        // set correct output index to one in vector
        correctOutputVector[correctOutput] = 1;

        // convert list of input values from strings to normalized doubles
        for (int i = 0; i < inputValues.length; i++)
            this.inputValues[i] = Double.parseDouble(inputValues[i]) / 255.0;
    }
}
