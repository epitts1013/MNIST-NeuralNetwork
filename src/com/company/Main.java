package com.company;
import java.io.*;
import java.util.Arrays;

public class Main
{
    // boolean determines whether file paths are gotten from command line or hardcoded into program
    static final boolean USE_CMD_ARGS = false;

    // hardcoded values for training and testing dataset size
    static final int TRAINING_DATA_SIZE = 60000, TESTING_DATA_SIZE = 10000;

    // boolean activates debug statements
    static final boolean DEBUG = false;

    public static void main(String[] args)
    {
        // store file paths to data files
        String trainingDataFilePath, testingDataFilePath;

        // store datasets
        NetworkInput[] trainingData = new NetworkInput[TRAINING_DATA_SIZE];
        NetworkInput[] testingData = new NetworkInput[TESTING_DATA_SIZE];

        // create neural network
        NeuralNetwork mnistNetwork = new NeuralNetwork(784, 1, 15, 10);

        // get input data file paths from command line arguments
        if (USE_CMD_ARGS)
        {
            trainingDataFilePath = args[0];
            testingDataFilePath = args[1];
        }
        else  // get input data file paths from hard-coded file paths
        {
            trainingDataFilePath = "C:\\Users\\super\\Documents\\Code\\MNIST-NeuralNetwork\\mnist_train.csv";
            testingDataFilePath = "C:\\Users\\super\\Documents\\Code\\MNIST-NeuralNetwork\\mnist_test.csv";
        }

        // create file objects for training and testing data
        File trainingDataFile = new File(trainingDataFilePath);
        File testingDataFile = new File(testingDataFilePath);

        // create BufferedReader for reading in training data
        BufferedReader bufferedReader;

        try
        {
            // attempt to initialize BufferedReader for training data
            bufferedReader = new BufferedReader(new FileReader(trainingDataFile));

            // variables for storing line information
            String line;
            String[] splitLine;

            // read data from training data file
            for (int i = 0; (line = bufferedReader.readLine()) != null; i++)
            {
                // separate each line into input data and correct output
                splitLine = line.split(",");
                trainingData[i] = new NetworkInput(Integer.parseInt(splitLine[0]), Arrays.copyOfRange(splitLine, 1, splitLine.length));
            }

            // attempt to initialize BufferedReader for testing data
            bufferedReader = new BufferedReader(new FileReader(testingDataFile));

            // read data from testing data file
            for (int i = 0; (line = bufferedReader.readLine()) != null; i++)
            {
                // separate each line into input data and correct output
                splitLine = line.split(",");
                testingData[i] = new NetworkInput(Integer.parseInt(splitLine[0]), Arrays.copyOfRange(splitLine, 1, splitLine.length));
            }

            // close buffered reader
            bufferedReader.close();
        }
        catch (FileNotFoundException e)  // the file path given by trainingDataFilePath was invalid
        {
            System.out.println("The specified file could not be found: \"" + trainingDataFilePath + "\"");
        }
        catch (IOException e)  // an error occurred during file reading
        {
            System.out.println("An error occurred while reading the training data file");
        }

        if (!DEBUG)
        {
            System.out.println("Beginning training");
            // run network training
            mnistNetwork.TrainNetwork(trainingData, 3.0, 10, 30);

            // test network on training data
            boolean[] correctOutputs = mnistNetwork.TestNetwork(trainingData);

            // calculate % of training data answered successfully
            int numCorrect = 0;
            for (boolean output : correctOutputs)
                if (output) numCorrect++;
            System.out.println("Network answered " + numCorrect + "/" + correctOutputs.length + " (" + (((double)numCorrect / (double)correctOutputs.length) * 100) + "%) of training cases correctly.");

            // test network on training data
            correctOutputs = mnistNetwork.TestNetwork(testingData);

            // calculate % of training data answered successfully
            numCorrect = 0;
            for (boolean output : correctOutputs)
                if (output) numCorrect++;
            System.out.println("Network answered " + numCorrect + "/" + correctOutputs.length + " (" + (((double)numCorrect / (double)correctOutputs.length) * 100) + "%) of testing cases correctly.");
        }

        // DEBUG
        if (DEBUG)
        {
            NeuralNetwork testNetwork = new NeuralNetwork(4, 1, 3, 2);
            testNetwork.LoadNetwork(new double[][]{{0.1, -0.36, -0.31}}, new double[][][]{{{-0.21, 0.72, -0.25, 1}, {-0.94, -0.41, -0.47, 0.63}, {0.15, 0.55, -0.49, -0.75}}}, new double[]{0.16, -0.46}, new double[][]{{0.76, 0.48, -0.73}, {0.34, 0.89, -0.23}});
            NetworkInput[] testInputs = new NetworkInput[4];
            testInputs[0] = new NetworkInput(1, "0,1,0,1".split(","));
            testInputs[1] = new NetworkInput(0, "1,0,1,0".split(","));
            testInputs[2] = new NetworkInput(1, "0,0,1,1".split(","));
            testInputs[3] = new NetworkInput(0, "1,1,0,0".split(","));
            testNetwork.TrainNetwork(testInputs, 10, 2, 6);
        }
    }
}
