package it.cnr.anomaly;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AnomalyDetectionVariationalAutoEncoder implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private static final Logger log = LoggerFactory.getLogger(AnomalyDetectionVariationalAutoEncoder.class);

	public double[] final_scores;
	public MultiLayerNetwork net;
	public double[] quantiles;
	public MatrixNormaliser mn = new MatrixNormaliser();
	
	public void train(double[][] featureMatrix, int nhidden, int nEpochs, File cache) throws Exception {

		// File cache = new File("vae_"+nhidden+"_"+nEpochs+".bin");

		int minibatchSize = featureMatrix.length;
		int rngSeed = 12345;
		// see An & Cho for details
		int nFeatures = featureMatrix[0].length;

		INDArray arr = Nd4j.create(featureMatrix);
		org.nd4j.linalg.dataset.DataSet d = new org.nd4j.linalg.dataset.DataSet(arr, arr);
		DataSetIterator trainIter = new ViewIterator(d, minibatchSize);

		// Neural net configuration
		Nd4j.getRandom().setSeed(rngSeed);

		// Neural net configuration
		// x->(LReLU)->h->h/2->z->(identity)->h/2->h->(sigmoid)->Bernoulli->p(x)
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed).updater(new Adam(1e-3))
				.weightInit(WeightInit.XAVIER).l2(1e-4).list()
				.layer(new VariationalAutoencoder.Builder().activation(Activation.LEAKYRELU).encoderLayerSizes(nhidden)
						.encoderLayerSizes(nhidden / 2) // 2 encoder layers
						.decoderLayerSizes(nhidden / 2).decoderLayerSizes(nhidden) // 2 decoder layers
						.pzxActivationFunction(Activation.IDENTITY) // p(z|data) activation function
						.reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID)) // Bernoulli
																													// reconstruction
																													// distribution
																													// +
																													// sigmoid
																													// activation
																													// -
																													// for
																													// modelling
																													// binary
																													// data
																													// (or
																													// data
																													// in
																													// range
																													// 0
																													// to
																													// 1)
						.nIn(nFeatures) // Input size
						.nOut(nFeatures).build()) // Size of the latent variable space: p(z|x)
				.build();

		// Train model:

		net = new MultiLayerNetwork(conf);
		net.init();
		log.warn(net.summary());

		System.out.println("Starting training...");
		for (int i = 0; i < nEpochs; i++) {
			net.pretrain(trainIter);
		}

		if (cache != null) {
			net.save(cache);
			System.out.println("Model cached to " + cache.getAbsolutePath());
		}

	}

	public static AnomalyDetectionVariationalAutoEncoder load(File cacheFile) throws Exception {
		
		File cacheFileMatrix = new File (cacheFile.getParent(),cacheFile.getName()+".mat");
		File cacheModel = new File (cacheFile.getParent(),cacheFile.getName()+".net");
		
		FileInputStream fileOut = new FileInputStream(cacheFileMatrix.getAbsolutePath());
		ObjectInputStream objectOut = new ObjectInputStream(fileOut);
		MatrixNormaliser mn = (MatrixNormaliser) objectOut.readObject();
	    objectOut.close();
	    
	    MultiLayerNetwork net = MultiLayerNetwork.load(cacheModel, false);
	    
	    fileOut = new FileInputStream(cacheFile.getAbsolutePath());
		objectOut = new ObjectInputStream(fileOut);
		AnomalyDetectionVariationalAutoEncoder vae = (AnomalyDetectionVariationalAutoEncoder) objectOut.readObject();
		objectOut.close();
	    
		vae.net = net;
		vae.mn = mn;
		
				//net = MultiLayerNetwork.load(vae, false);
		/*
		FileInputStream fileOut = new FileInputStream(vae.getAbsolutePath());
	    ObjectInputStream objectOut = new ObjectInputStream(fileOut);
	    Object o = objectOut.readObject();
	    
	    AnomalyDetectionVariationalAutoEncoder mn = (AnomalyDetectionVariationalAutoEncoder) o;  // Serialize and write object to file
        System.out.println("The VAE was successfully read from " + vae.getAbsolutePath());
        objectOut.close();
        */
        return vae;
	}

	public static void save(AnomalyDetectionVariationalAutoEncoder vae, File cacheFile) throws Exception{
			File cacheFileMatrix = new File (cacheFile.getParent(),cacheFile.getName()+".mat");
			File cacheModel = new File (cacheFile.getParent(),cacheFile.getName()+".net");
			
		    FileOutputStream fileOut = new FileOutputStream(cacheFileMatrix.getAbsolutePath());
			ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
			objectOut.writeObject(vae.mn);
		    objectOut.close();
		    
		    vae.net.save(cacheModel);
		    
		    vae.net = null;
		    vae.mn = null;
		    fileOut = new FileOutputStream(cacheFile.getAbsolutePath());
			objectOut = new ObjectOutputStream(fileOut);
			objectOut.writeObject(vae);
		    objectOut.close();
			
		    System.out.println("VAE (+mat and net files) successfully written to " + cacheFile.getAbsolutePath());
		    
	}
	
	public double[] test1by1(double[][] featureMatrix, int reconstructionNumSamples) throws Exception {

		org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net
				.getLayer(0);
		final_scores = new double[featureMatrix.length];
		System.out.println("Features vs Score");
		for (int j = 0; j < featureMatrix.length; j++) {
			double[][] featureMatrix1 = MathOperations.subsetRows(featureMatrix, j, j);
			INDArray arr1 = Nd4j.create(featureMatrix1);
			org.nd4j.linalg.dataset.DataSet d1 = new org.nd4j.linalg.dataset.DataSet(arr1, arr1);
			DataSetIterator trainIter1 = new ViewIterator(d1, featureMatrix1.length);
			DataSet ds1 = trainIter1.next();
			INDArray features1 = ds1.getFeatures();
			INDArray reconstructionErrorEachExample1 = vae.reconstructionLogProbability(features1,
					reconstructionNumSamples);
			double final_score = reconstructionErrorEachExample1.getDouble(0);
			final_scores[j] = final_score;
			//System.out.println(features1.getDouble(0)+","+featureMatrix1[0][0]+","+final_scores[j]);
		}
		
		System.out.println("");
		return final_scores;
	}

	public double[] test(double[][] featureMatrix, int reconstructionNumSamples) throws Exception {

		org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net
				.getLayer(0);

		INDArray arr = Nd4j.create(featureMatrix);
		org.nd4j.linalg.dataset.DataSet d = new org.nd4j.linalg.dataset.DataSet(arr, arr);
		DataSetIterator trainIter = new ViewIterator(d, featureMatrix.length);
		DataSet ds = trainIter.next();
		INDArray features = ds.getFeatures();
		INDArray reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples);

		int nRows = features.rows();
		final_scores = new double[featureMatrix.length];

		for (int j = 0; j < nRows; j++) {
			final_scores[j] = reconstructionErrorEachExample.getDouble(j);
		}
		
		return final_scores;
	}

	public String[] classify() {

		if (quantiles == null)
			quantiles = MathOperations.quantiles(final_scores);
		String[] classification = new String[final_scores.length];
		int i = 0;
		for (double f : final_scores) {
			if (f <= quantiles[0]) {
				classification[i] = "0p-25p";
			} else if (f <= quantiles[1]) {
				classification[i] = "25p-50p";
			} else if (f <= quantiles[2]) {
				classification[i] = "50p-75p";
			} else {
				classification[i] = "75p-100p";
			}

			i++;
		}
		return classification;

	}

	// binarises a matrix based on thresholds per column
	public static double[][] binariseMatrix(double[][] matrix, double[] binarisationThresholds) {
		int nrow = matrix.length;
		int ncol = matrix[0].length;
		double[][] binarymatrix = new double[nrow][ncol];

		for (int i = 0; i < nrow; i++) {
			for (int j = 0; j < ncol; j++) {
				if (matrix[i][j] > binarisationThresholds[j]) {
					binarymatrix[i][j] = 1;
				} else
					binarymatrix[i][j] = 0;
			}
		}

		return binarymatrix;
	}

}
