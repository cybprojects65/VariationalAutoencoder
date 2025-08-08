package it.cnr.anomaly;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;

public class MatrixNormaliser implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public static void main(String[] args) throws Exception{
		
		
		String filePath = "Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv";
		String arguments[] = {"environment 2017_land_distance","environment 2017_mean_depth"};
		
		double [][] features = CSV2Features.readFeatures(filePath,arguments);
		System.out.println("Read "+features.length+" X "+features[0].length+" features");
		
		String encoded = MathOperations.encodeStrings(arguments);
		System.out.println("Encoded: "+encoded);
		
		File cache = new File("matrix_norm_"+features.length+"X"+features[0].length+"_"+encoded+".bin");
		
		MatrixNormaliser mn = new MatrixNormaliser();
		double [][] features01 =  mn.normalize(features,cache);
		
		System.out.println("Feature sample 1 "+Arrays.toString(features01[0]));
		
		MatrixNormaliser mn2 = MatrixNormaliser.load(cache);
		
		double [][] features02 = mn2.renormalize(features);
		
		System.out.println("Feature sample 2 "+Arrays.toString(features02[0]));
	}

	double [] means;
	double [] sds;
	double [] mins;
	double [] maxs;
	
	public double [][] normalize(double [][] features, File cacheFile) throws Exception{
		
		MathOperations math = new MathOperations();
		double [][] featuresStd = math.standardize(features);
		means = math.means;
		sds = math.sds;
		double [][] features01 = math.normalize01(featuresStd);
		mins = math.mins;
		maxs = math.maxs;
		
		if (cacheFile!=null) {
			 try (FileOutputStream fileOut = new FileOutputStream(cacheFile.getAbsolutePath());
		             ObjectOutputStream objectOut = new ObjectOutputStream(fileOut)) {

		            objectOut.writeObject(this);  // Serialize and write object to file
		            System.out.println("The object was successfully written to " + cacheFile.getAbsolutePath());

		        } catch (IOException e) {
		            System.err.println("Error saving object: " + e.getMessage());
		            e.printStackTrace();
		        }
		}
		return features01;
	}
	
	public double [][] renormalize(double [][] features) throws Exception{
		
		MathOperations math = new MathOperations();
		double [][] featuresStd = math.standardize(features, means, sds);
		double [][] features01 = math.normalize01(featuresStd, mins, maxs);
		
		return features01;
	}
	
	public double [][] denormalize_destandardize(double [][] features) throws Exception{
	
		MathOperations math = new MathOperations();
		
		double [][] features01 = math.denormalize01(features, mins, maxs);
		double [][] featuresDeStd = math.destandardize(features01, means, sds);
		
		
		return featuresDeStd;
	}

	public static MatrixNormaliser load(File cacheFile) throws Exception{
		
		FileInputStream fileOut = new FileInputStream(cacheFile.getAbsolutePath());
	    ObjectInputStream objectOut = new ObjectInputStream(fileOut);
        MatrixNormaliser mn = (MatrixNormaliser) objectOut.readObject();  // Serialize and write object to file
        System.out.println("The object was successfully read from " + cacheFile.getAbsolutePath());
        return mn;
        
	}
}
