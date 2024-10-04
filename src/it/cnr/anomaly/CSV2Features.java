package it.cnr.anomaly;

import com.opencsv.CSVReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CSV2Features {


	public static double[][] readFeatures(String filePath, String... args) throws Exception{

		List<String> columns2Select = new ArrayList<String>();

		if (args.length > 0) {
			for (int i = 0; i < args.length; i++) {
				columns2Select.add(args[i]);
			}
		}

		try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
			List<String[]> rows = reader.readAll();
			// Specify the column indices you want to read
			int[] columnsToRead = new int[columns2Select.size()]; // Reading column 0 and column 2
			double [][] features = new double [rows.size()-1] [columnsToRead.length];
			int rowCounter = 0;
			for (String[] row : rows) {
				if (rowCounter == 0) {
					boolean featuresfound = false;
					int columnCounter = 0;
					int elementCounter = 0;
					for (String r : row) {
						if (columns2Select.contains(r)) {
							columnsToRead[elementCounter] = columnCounter;
							elementCounter++;
							featuresfound = true;
							System.out.println("Found feature: "+r);
						}
						columnCounter++;
					}
					if (!featuresfound)
						throw new Exception("No column corresponds to those provided");
				} else {
					int elementCounter = 0;
					for (int col : columnsToRead) {
						if (col < row.length) {
							try {
							features[rowCounter-1][elementCounter]=Double.parseDouble(row[col]);
							}catch(Exception ee) {
								ee.printStackTrace();
								System.out.println("Error with variable "+columns2Select.get(elementCounter)+"(column:"+col+"): wrong number at "+row[col]);
							}
							elementCounter++;
						}
						
					}
				}
				
				rowCounter++;
			}
			return features;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static void main(String[] args) throws Exception{
		
		String filePath = "Complete_dataset_mediterranean_sea_2017_2018_2019_2020_2021_2050RCP8.5.csv";
		String arguments[] = {"environment 2017_land_distance","environment 2017_mean_depth"};
		
		double [][] features = readFeatures(filePath,arguments);
		System.out.println("Read "+features.length+" X "+features[0].length+" features");
	}
	
}
