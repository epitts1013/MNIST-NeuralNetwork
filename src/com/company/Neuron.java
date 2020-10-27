package com.company;
import java.lang.Math;

public class Neuron
{
    // sets whether neuron is an input neuron, if neuron is an input neuron its output
    // can be set manually but its output cannot be calculated based on inputs, and
    // in fact does not have any inputs or bias
    final boolean INPUT_NEURON;

    // class variables
    private double bias, activation;
    private double[] weights;
    private final Neuron[] inputs;

    // gradient variables
    private double biasGradient;
    private double sumBiasGradients;
    private double[] weightGradient;
    private double[] sumWeightGradients;

    // Constructor for neuron used for network input
    public Neuron()
    {
        // set neuron as input neuron
        INPUT_NEURON = true;

        // initialize variable as default values
        bias = 0;
        inputs = null;
        weights = null;
        biasGradient = 0;
        sumBiasGradients = 0;
        weightGradient = null;
        sumWeightGradients = null;
    }

    // Constructor for neuron with neuron outputs as its inputs
    public Neuron(Neuron[] inputs)
    {
        // set neuron as not input neuron
        INPUT_NEURON = false;

        // initialize variables
        bias = (Math.random() > 0.5) ? Math.random() : -Math.random();
        this.inputs = inputs;
        weights = new double[inputs.length];
        biasGradient = 0;
        sumBiasGradients = 0;
        weightGradient = new double[inputs.length];
        sumWeightGradients = new double[inputs.length];

        // generate random initial weights
        for (int i = 0; i < weights.length; i++)
            weights[i] = (Math.random() > 0.5) ? Math.random() : -Math.random();

        // initialize values of weightGradient and sumWeightGradient
        for (int i = 0; i < weightGradient.length; i++)
        {
            weightGradient[i] = 0;
            sumWeightGradients[i] = 0;
        }
    }

    // Calculates output of neuron
    // If neuron is initialized as an input neuron, does nothing
    public void CalculateActivation()
    {
        // if not in input neuron, proceed as normal
        if (!INPUT_NEURON)
        {
            // zero out output
            double sum = 0;

            // perform dot-product
            for (int i = 0; i < weights.length; i++)
                sum += inputs[i].GetActivation() * weights[i];

            // add bias to sum
            sum += bias;

            // use sum as input for sigmoid function
            activation = 1.0 / (1.0 + Math.pow(Math.E, -sum));
        }
        else
            System.out.println("Activation cannot be calculated for input neuron");
    }

    // applies the weight and bias gradient sums to the weights and bias
    public void ApplyGradients(double learnRate, int batchSize)
    {
        // apply bias gradient
        bias = bias - (learnRate / batchSize * sumBiasGradients);

        // apply weight gradient
        for (int i = 0; i < sumWeightGradients.length; i++)
            weights[i] = weights[i] - ((learnRate / batchSize) * sumWeightGradients[i]);

        // zero out gradient variables
        biasGradient = 0;
        sumBiasGradients = 0;

        for (int i = 0; i < weightGradient.length; i++)
        {
            weightGradient[i] = 0;
            sumWeightGradients[i] = 0;
        }
    }

    // region GettersAndSetters
    // getter for output
    public double GetActivation()
    {
        return activation;
    }

    // setter for output, does not work if INPUT_NEURON is not true
    public void SetActivation(double value)
    {
        if (INPUT_NEURON)
            activation = value;
        else
            System.out.println("Value for neuron cannot be manually set if it is not initialized as an input neuron");
    }

    // getter for inputs
    public Neuron[] GetInputs()
    {
        return inputs;
    }

    // getter for bias
    public double GetBias()
    {
        return bias;
    }

    // setter for bias, used in loading network
    public void SetBias(double bias)
    {
        this.bias = bias;
    }

    // getter for weights
    public double[] GetWeights()
    {
        return weights;
    }

    // setter for weights, used for loading trained network
    public void SetWeights(double[] weights)
    {
        this.weights = weights;
    }

    // getter for bias gradient
    public double GetBiasGradient()
    {
        return biasGradient;
    }

    // setter for bias gradient
    public void SetBiasGradient(double biasGradient)
    {
        this.biasGradient = biasGradient;
        sumBiasGradients += biasGradient;
    }

    // getter for weight gradient
    public double[] GetWeightGradient()
    {
        return weightGradient;
    }

    // setter for weight gradient
    public void SetWeightGradient(double[] weightGradient)
    {
        this.weightGradient = weightGradient;
        sumWeightGradients = AddArrays(this.sumWeightGradients, weightGradient);
    }
    // endregion

    // helper methods
    private double[] AddArrays(double[] a, double[] b)
    {
        if (a.length == b.length)
        {
            double[] out = new double[a.length];

            for (int i = 0; i < out.length; i++)
            {
                out[i] = a[i] + b[i];
            }

            return out;
        }
        else
        {
            System.out.println("Arrays must be same size to add");
            return null;
        }
    }

    private double[] SubArrays(double[] a, double[] b)
    {
        if (a.length == b.length)
        {
            double[] out = new double[a.length];

            for (int i = 0; i < out.length; i++)
            {
                out[i] = a[i] - b[i];
            }

            return out;
        }
        else
        {
            System.out.println("Arrays must be same size to subtract");
            return null;
        }
    }
}
