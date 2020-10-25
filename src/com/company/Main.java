package com.company;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

public class Main
{
    // boolean determines whether filepaths are gotten from command line or hardcoded into program
    static final boolean USE_CMD_ARGS = true;

    // hardcoded values for training and testing dataset size
    static final int TRAINING_DATA_SIZE = 60000, TESTING_DATA_SIZE = 10000;

    public static void main(String[] args)
    {
        // store file paths to data files
        String trainingDataFilePath, testingDataFilePath;

        // store datasets
        NetworkInput[] trainingData = new NetworkInput[TRAINING_DATA_SIZE];
        NetworkInput[] testingData = new NetworkInput[TESTING_DATA_SIZE];

        // get input data filepaths from command line arguments
        if (USE_CMD_ARGS)
        {
            trainingDataFilePath = args[0];
            testingDataFilePath = args[1];
        }
        else  // get input data filepaths from hard-coded filepaths
        {
            trainingDataFilePath = "{some path}/mnist_train.csv";
            testingDataFilePath = "{some path}/mnist_test.csv";
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
        catch (IOException e)  // an error occured during file reading
        {
            System.out.println("An error occured while reading the training data file");
        }

        // create neural network
        NeuralNetwork mnistNetwork = new NeuralNetwork(784, 1, 15, 10);

        // run network through training data
        for (NetworkInput inputData : trainingData)
        {
            // input data
            mnistNetwork.InputNetworkData(inputData);

            // TODO: Set up training methods for neural network
        }
    }
}
