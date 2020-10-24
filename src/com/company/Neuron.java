package com.company;
import java.lang.Math;

public class Neuron
{
    // class variables
    private double bias, output;
    private double[] weights;
    private final Neuron[] inputs;

    // constructor for neuron with a double array as inputs
//    public Neuron(int numInputs)
//    {
//        // initialize variables
//        bias = Math.random();
//        inputs = null;
//        weights = new double[numInputs];
//
//        // generate random initial weights
//        for (int i = 0; i < weights.length; i++)
//            weights[i] = Math.random();
//    }

    // Constructor for neuron with neuron outputs as its inputs
    public Neuron(Neuron[] inputs)
    {
        // initialize variables
        bias = Math.random();
        this.inputs = inputs;
        this.weights = new double[inputs.length];

        // generate random initial weights
        for (int i = 0; i < weights.length; i++)
            weights[i] = Math.random();
    }

    // version of CalculateOutput for use with neurons that recieve input from other neurons (i.e. all layers besides first layer)
    public double CalculateOutput()
    {
        // zero out output
        double sum = 0;

        try
        {
            // perform dot-product
            for (int i = 0; i < weights.length; i++)
                sum += inputs[i].GetOutput() * weights[i];
        }
        catch (ArrayIndexOutOfBoundsException e)  // tried to run normal output calculation on neuron initialized for direct input
        {
            System.out.println("Neuron was initialized for direct input, please use overloaded method CalculateOutput(double[] directInputs)");
        }

        // add bias to sum
        sum += bias;

        // use sum as input for sigmoid function
        output = 1 / (1 + Math.pow(Math.E, -sum));

        // return output
        return output;
    }

    // version of CalculateOutput for use with neurons that recieve direct input (i.e. first layer neurons)
    public double CalculateOutput(double[] directInputs)
    {
        // zero out output
        double sum = 0;

        try
        {
            // perform dot-product
            for (int i = 0; i < weights.length; i++)
                sum += directInputs[i] * weights[i];
        }
        catch (ArrayIndexOutOfBoundsException e)  // length of directInputs does not match length of weights
        {
            System.out.println("Length of double array must match number of weights");
        }

        // add bias to sum
        sum += bias;

        // use sum as input for sigmoid function
        output = 1 / (1 + Math.pow(Math.E, -sum));

        // return output
        return output;
    }

    // getter for output
    public double GetOutput()
    {
        return output;
    }
}
