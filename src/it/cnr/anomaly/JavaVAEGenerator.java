package it.cnr.anomaly;

import java.io.File;
import java.util.Arrays;

public class JavaVAEGenerator extends JavaVAE {
	static boolean debug = false;
	public static void main(String[] args) throws Exception{

		System.out.println("VAE started");
		
		JavaVAEGenerator jvae = new JavaVAEGenerator();
		
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
		File classificationFile = new File(outputFolder,"reconstruction.csv");
		
		AnomalyDetectionVariationalAutoEncoder vae = new AnomalyDetectionVariationalAutoEncoder();
		double[][] features01 = vae.mn.normalize(features, null);
		vae.train4Reconstruction(features01, nHidden, nEpochs, null);
		vae.generate(features01, reconstructionNumSamples);
		String [] classifications = vae.classify();
		AnomalyDetectionVariationalAutoEncoder.save(vae,cacheModel);
		//double [][] features, double [][] reconstructedfeatures, String[] headers, double []scores, double []probabilities, String[] percentiles, File outputFile
		
		saveReconstruction(features, vae.reconstructed_features, variables, vae.final_scores, vae.reconstruction_scores ,classifications, classificationFile);
		
	}

}
