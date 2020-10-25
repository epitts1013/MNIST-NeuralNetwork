package com.company;
import java.lang.Math;

public class Neuron
{
    // sets whether neuron is an input neuron, if neuron is an input neuron its output can be set manually
    final boolean INPUT_NEURON;

    // class variables
    private double bias, output;
    private double[] weights;
    private final Neuron[] inputs;

    // Constructor for neuron used for network input
    public Neuron()
    {
        // set neuron as input neuron
        INPUT_NEURON = true;

        // initialize variable as default values
        bias = 0;
        inputs = null;
        weights = null;
    }

    // Constructor for neuron with neuron outputs as its inputs
    public Neuron(Neuron[] inputs)
    {
        // set neuron as not input neuron
        INPUT_NEURON = false;

        // initialize variables
        bias = Math.random();
        this.inputs = inputs;
        this.weights = new double[inputs.length];

        // generate random initial weights
        for (int i = 0; i < weights.length; i++)
            weights[i] = Math.random();
    }

    // Calculates output of neuron
    // If neuron is initialized as an input neuron, just returns output
    // Yes, there's already a GetOutput(), I need it for other methods, I don't want errors if
    // somebody tries to use this method on an input neuron
    public double CalculateOutput()
    {
        // if not in input neuron, proceed as normal
        if (!INPUT_NEURON)
        {
            // zero out output
            double sum = 0;

            // perform dot-product
            for (int i = 0; i < weights.length; i++)
                sum += inputs[i].GetOutput() * weights[i];

            // add bias to sum
            sum += bias;

            // use sum as input for sigmoid function
            output = 1 / (1 + Math.pow(Math.E, -sum));
        }

        // return output
        return output;
    }

    // getter for output
    public double GetOutput()
    {
        return output;
    }

    // setter for output, does not work if INPUT_NEURON is not true
    public void SetOutput(double value)
    {
        if (INPUT_NEURON)
            output = value;
        else
            System.out.println("Value for neuron cannot be manually set if it is not initialized as an input neuron");
    }
}
