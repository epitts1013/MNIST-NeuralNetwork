package com.company;

public class NetworkInput
{
    public int correctOutput;
    public double[] inputValues;

    public NetworkInput(int correctOutput, String[] inputValues)
    {
        // initialize variables
        this.correctOutput = correctOutput;
        this.inputValues = new double[inputValues.length];

        // convert list of input values from strings to normalized doubles
        for (int i = 0; i < inputValues.length; i++)
            this.inputValues[i] = Double.parseDouble(inputValues[i]) / 255.0;
    }
}
