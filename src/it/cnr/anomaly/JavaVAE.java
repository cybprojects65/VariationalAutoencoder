package it.cnr.anomaly;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;

public class JavaVAE {
	/*
	 * public static boolean useDLClassification = true; public static int nEpochs =
	 * 1000; public static int reconstructionNumSamples = 16; // fixed public static
	 * int nhidden = 8; public static int nClusters4AnomalyDetection = 5;
	 * 
	 */
	
	boolean train = true;
	File filePath = null;
	int nHidden = -1;
	int nEpochs = -1;
	File outputFolder = null;
	int reconstructionNumSamples  = -1;
	File modelFile = null;
	File matrixFile = null;
	String [] variables = null;
	
	public static void printNotes() {
		System.out.println("Parameters:");
		System.out.println("-i: input file path");
		System.out.println("-v: variable names (columns) separated by commas");
		System.out.println("-h: number of hidden nodes");
		System.out.println("-e: number of epochs");
		System.out.println("-o: output folder");
		System.out.println("-r: number of reconstruction samples");
		System.out.println("-m: trained model file (for projections)");
		System.out.println("-t: training mode active (true/false)");
		
	}
	public static void main(String[] args) throws Exception{

		System.out.println("VAE started");
		
		JavaVAE jvae = new JavaVAE();
		
		if (args==null || args.length==0)
		{
			printNotes();
			System.exit(2);
		}
		for (String a: args) {
			
			if (a.startsWith("-t")) {
				jvae.train = Boolean.parseBoolean(a.substring(2).trim());
			}
			else if (a.startsWith("-i")) {
				jvae.filePath = new File(a.substring(2).trim().replace("\"", ""));
			}
			else if (a.startsWith("-h")) {
				jvae.nHidden = Integer.parseInt(a.substring(2).trim());
			}
			else if (a.startsWith("-e")) {
				jvae.nEpochs = Integer.parseInt(a.substring(2).trim());
			}
			else if (a.startsWith("-o")) {
				jvae.outputFolder = new File(a.substring(2).trim());
			}
			else if (a.startsWith("-r")) {
				jvae.reconstructionNumSamples = Integer.parseInt(a.substring(2).trim());
			}
			else if (a.startsWith("-m")) {
				jvae.modelFile = new File(a.substring(2).trim());
			}
			else if (a.startsWith("-v")) {
				String variableS = a.substring(2).replace("\"", "").replace("'", "");
				jvae.variables = variableS.split(",");
			}
			
		}
		
		
		if (jvae.train) {
			if (jvae.filePath==null) {
				System.out.println("ERROR: Input file not provided with the -i option");
				printNotes();
				System.exit(2);
			}
			if (jvae.nHidden==-1) {
				System.out.println("ERROR: Number of hidden layers not provided with the -h option");
				printNotes();
				System.exit(2);
			}
			if (jvae.nEpochs==-1) {
				System.out.println("ERROR: Number of epochs not provided with the -e option");
				printNotes();
				System.exit(2);
			}
			if (jvae.outputFolder==null) {
				System.out.println("ERROR: Output folder not provided with the -o option");
				printNotes();
				System.exit(2);
			}
			if (jvae.reconstructionNumSamples==-1) {
				System.out.println("ERROR: Number of reconstruction samples not provided with the -r option");
				printNotes();
				System.exit(2);
			}
			if (jvae.variables==null) {
				System.out.println("ERROR: variables not provided with the -v option");
				printNotes();
				System.exit(2);
			}
			System.out.println("input file: "+jvae.filePath.getAbsolutePath());
			System.out.println("nHidden: "+jvae.nHidden);
			System.out.println("nEpochs: "+jvae.nEpochs);
			System.out.println("outputFolder: "+jvae.outputFolder.getAbsolutePath());
			System.out.println("reconstructionNumSamples: "+jvae.reconstructionNumSamples);
			System.out.println("variables: "+Arrays.toString(jvae.variables));
			
			
			jvae.trainVAE();
		}else {
			if (jvae.filePath==null) {
				System.out.println("ERROR: Input file not provided with the -i option");
				printNotes();
				System.exit(2);
			}
			if (jvae.modelFile==null) {
				System.out.println("ERROR: Model file not provided with the -m option");
				printNotes();
				System.exit(2);
			}
			if (jvae.outputFolder==null) {
				System.out.println("ERROR: Output folder not provided with the -o option");
				printNotes();
				System.exit(2);
			}
			if (jvae.reconstructionNumSamples==-1) {
				System.out.println("ERROR: Number of reconstruction samples not provided with the -r option");
				printNotes();
				System.exit(2);
			}
			if (jvae.variables==null) {
				System.out.println("ERROR: variables not provided with the -v option");
				printNotes();
				System.exit(2);
			}
			System.out.println("input file: "+jvae.filePath.getAbsolutePath());
			System.out.println("nHidden: "+jvae.nHidden);
			System.out.println("nEpochs: "+jvae.nEpochs);
			System.out.println("outputFolder: "+jvae.outputFolder.getAbsolutePath());
			System.out.println("reconstructionNumSamples: "+jvae.reconstructionNumSamples);
			System.out.println("variables: "+Arrays.toString(jvae.variables));
			
			jvae.testVAE();
		}
	}

	static boolean debug = false;
	public void trainVAE() throws Exception {

		double[][] features = CSV2Features.readFeatures(filePath.getAbsolutePath(), variables);

		if (debug) {
			double [][] featuresmall = new double [100][variables.length];
					
			for (int i=0;i<featuresmall.length;i++) {
				for (int j=0;j<featuresmall[0].length;j++) {
					featuresmall[i][j] = features[i][j];
				}
			}	
			features = featuresmall;
		}
		
		if (!outputFolder.exists())
			outputFolder.mkdir();
		
		System.out.println("Read " + features.length + " X " + features[0].length + " features");
		String encoded = MathOperations.encodeStrings(variables);

		System.out.println("Encoded: " + encoded);
		//File cacheModel = new File(outputFolder, "model_norm_" + features.length + "X" + features[0].length + "_" + encoded + ".bin");
		//File classificationFile = new File(outputFolder, 
			//	"classification_" + features.length + "X" + features[0].length + "_" + encoded + ".csv");
		
		File cacheModel = new File(outputFolder, "model.bin");
		File classificationFile = new File(outputFolder,"classification.csv");
		
		AnomalyDetectionVariationalAutoEncoder vae = new AnomalyDetectionVariationalAutoEncoder();
		double[][] features01 = vae.mn.normalize(features, null);
		vae.train(features01, nHidden, nEpochs, null);
		vae.test(features01, reconstructionNumSamples);
		String [] classifications = vae.classify();
		AnomalyDetectionVariationalAutoEncoder.save(vae,cacheModel);
		saveClassification(features,variables, vae.final_scores, classifications, classificationFile);
		
	}

	public void testVAE() throws Exception {

		double[][] features = CSV2Features.readFeatures(filePath.getAbsolutePath(), variables);
		
		AnomalyDetectionVariationalAutoEncoder vae = AnomalyDetectionVariationalAutoEncoder.load(modelFile);
		double[][] features01 = vae.mn.renormalize(features);
		vae.test(features01, reconstructionNumSamples);
		String encoded = MathOperations.encodeStrings(variables);
		File classificationFile = new File(outputFolder,"classification_projection.csv");

		String [] classifications = vae.classify();
		saveClassification(features,variables, vae.final_scores, classifications, classificationFile);
	}
	
	
	public void saveClassification(double [][] features, String[] headers, double []scores, String[] percentiles, File outputFile){
		
		try {
		BufferedWriter bf = new BufferedWriter(new FileWriter(outputFile));
				
		System.out.println("Saving results to file");
		String head = Arrays.toString(headers).replace("[", "").replace("]", "").replaceAll(" +", " ");
		head += ",reconstruction_log_probability,percentile";
		bf.append(head+"\n");
		
		for (int i=0;i<features.length;i++) {
			
			for (int j=0;j<features[0].length;j++) {
				bf.append(features[i][j]+",");
			}
			bf.append(scores[i]+","+percentiles[i]);
			//System.out.println(features[i][0]+","+scores[i]);
			if (i<(features.length-1))
				bf.append("\n");
		}
		bf.close();
		System.out.println("Scores and output written to "+outputFile.getAbsolutePath());
		}catch(Exception e) {
			
			e.printStackTrace();
			System.out.println("Error with the output file "+e.getLocalizedMessage());
			
		}
		 
	}

	public void saveReconstruction(double [][] features, double [][] reconstructedfeatures, String[] headers, double []scores, double []probabilities, String[] percentiles, File outputFile){
		
		try {
		BufferedWriter bf = new BufferedWriter(new FileWriter(outputFile));
				
		System.out.println("Saving results to file");
		String head = Arrays.toString(headers).replace("[", "").replace("]", "").replaceAll(" +", " ");
		String headRec = ",";
		for (int i = 0;i<headers.length;i++) {
			
			headRec +=headers[i]+"_rec,";
					
		}
		
		head += headRec+"probability,reconstruction_log_probability,percentile";
		bf.append(head+"\n");
		
		for (int i=0;i<features.length;i++) {
			
			for (int j=0;j<features[0].length;j++) {
				bf.append(features[i][j]+",");
			}
			for (int j=0;j<features[0].length;j++) {
				bf.append(reconstructedfeatures[i][j]+",");
			}
			bf.append(probabilities[i]+","+scores[i]+","+percentiles[i]);
			//System.out.println(features[i][0]+","+scores[i]);
			if (i<(features.length-1))
				bf.append("\n");
		}
		bf.close();
		System.out.println("Scores and output written to "+outputFile.getAbsolutePath());
		}catch(Exception e) {
			
			e.printStackTrace();
			System.out.println("Error with the output file "+e.getLocalizedMessage());
			
		}
		 
	}

}
