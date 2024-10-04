package it.cnr.anomaly;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;

public class MathOperations {

	// ###########Matrix/Vector manipulation
	public static double[][] subsetRows(double[][] matrix, int row0, int row1) {

		double[][] sub = new double[row1 - row0 + 1][matrix.length];

		for (int i = row0; i <= row1; i++) {

			sub[i - row0] = matrix[i];

		}

		return sub;

	}

	public static double[] columnToVector(double[][] matrix, int i) {

		double[] vector = new double[matrix.length];
		for (int k = 0; k < matrix.length; k++) {
			vector[k] = matrix[k][i];
		}
		return vector;
	}

	public static double columnMean(double[][] matrix, int i) {

		double[] vector = columnToVector(matrix, i);
		return mean(vector);
	}

	public static double columnMin(double[][] matrix, int i) {

		double[] vector = columnToVector(matrix, i);
		return min(vector);
	}

	public static double min(double[] values) {
		int n = values.length;
		if (n == 0) {
			throw new IllegalArgumentException("The input array is empty.");
		}

		// Initialize the minimum value to the first element
		double min = values[0];

		// Iterate through the array and update the minimum value
		for (int i = 1; i < n; i++) {
			if (values[i] < min) {
				min = values[i];
			}
		}

		return min;
	}

	public static double columnMax(double[][] matrix, int i) {

		double[] vector = columnToVector(matrix, i);
		return max(vector);
	}

	public static double max(double[] values) {
		int n = values.length;
		if (n == 0) {
			throw new IllegalArgumentException("The input array is empty.");
		}

		// Initialize the minimum value to the first element
		double max = values[0];

		// Iterate through the array and update the minimum value
		for (int i = 1; i < n; i++) {
			if (values[i] > max) {
				max = values[i];
			}
		}

		return max;
	}

	public static double columnSD(double[][] matrix, int i) {

		double[] vector = columnToVector(matrix, i);
		double variance = calculateVariance(vector);
		return Math.sqrt(variance);
	}

	public static double calculateVariance(double[] values) {
		int n = values.length;
		if (n == 0) {
			throw new IllegalArgumentException("The input array is empty.");
		}

		// Step 1: Calculate the mean
		double sum = 0.0;
		for (double value : values) {
			sum += value;
		}
		double mean = sum / n;

		// Step 2: Calculate the sum of squared differences from the mean
		double squaredDiffSum = 0.0;
		for (double value : values) {
			squaredDiffSum += (value - mean) * (value - mean);
		}

		// Step 3: Calculate variance (Population variance)
		return squaredDiffSum / n;
	}

	public static double[] columnQ1(double[][] matrix) {

		int ncol = matrix[0].length;
		double[] o = new double[ncol];

		for (int j = 0; j < ncol; j++) {
			double[] vector = columnToVector(matrix, j);
			double q1 = quantiles(vector)[0];
			o[j] = q1;
		}
		return o;
	}

	public static double[] columnQ3(double[][] matrix) {

		int ncol = matrix[0].length;
		double[] o = new double[ncol];

		for (int j = 0; j < ncol; j++) {
			double[] vector = columnToVector(matrix, j);
			double q3 = quantiles(vector)[2];
			o[j] = q3;
		}
		return o;
	}

	public static double[] columnMeans(double[][] matrix) {

		int ncol = matrix[0].length;
		double[] colmeans = new double[ncol];

		for (int j = 0; j < ncol; j++) {

			double mean = columnMean(matrix, j);
			colmeans[j] = mean;
		}
		return colmeans;
	}

	public static double[] columnSDs(double[][] matrix) {

		int ncol = matrix[0].length;
		double[] colsds = new double[ncol];

		for (int j = 0; j < ncol; j++) {
			double sd = columnSD(matrix, j);
			colsds[j] = sd;
		}

		return colsds;
	}

	public static ArrayList<Integer> generateRandoms(int numberOfRandoms, int min, int max) {

		ArrayList<Integer> randomsSet = new ArrayList<Integer>();
		// if number of randoms is equal to -1 generate all numbers
		if (numberOfRandoms == -1) {
			for (int i = min; i < max; i++) {
				randomsSet.add(i);
			}
		} else {
			int numofrandstogenerate = 0;
			if (numberOfRandoms <= max) {
				numofrandstogenerate = numberOfRandoms;
			} else {
				numofrandstogenerate = max;
			}

			if (numofrandstogenerate == 0) {
				randomsSet.add(0);
			} else {
				for (int i = 0; i < numofrandstogenerate; i++) {

					int RNum = -1;
					RNum = (int) ((max) * Math.random()) + min;

					// generate random number
					while (randomsSet.contains(RNum)) {
						RNum = (int) ((max) * Math.random()) + min;
						// AnalysisLogger.getLogger().debug("generated " + RNum);
					}

					// AnalysisLogger.getLogger().debug("generated " + RNum);

					if (RNum >= 0)
						randomsSet.add(RNum);
				}

			}
		}

		return randomsSet;
	}

	public static int[] generateSequence(int elements) {
		int[] sequence = new int[elements];
		for (int i = 0; i < elements; i++) {
			sequence[i] = i;
		}
		return sequence;
	}

	// searches for an index into an array
	public static boolean isIn(List<Integer> indexarray, int index) {

		int size = indexarray.size();

		for (int i = 0; i < size; i++) {
			if (index == indexarray.get(i).intValue())
				return true;
		}

		return false;
	}

	// finds the indexes of zero points
	public static List<Integer> findZeros(double[] points) {

		int size = points.length;
		ArrayList<Integer> zeros = new ArrayList<Integer>();

		for (int i = 0; i < size; i++) {
			if (points[i] == 0) {
				int start = i;
				int end = i;

				for (int j = i + 1; j < size; j++) {
					if (points[j] != 0) {
						end = j - 1;
						break;
					}
				}
				int center = start + ((end - start) / 2);
				zeros.add(center);
				i = end;
			}
		}

		return zeros;

	}

	public static double getArgMax(double[] points) {
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (max < points[i])
				max = points[i];
		}
		return max;
	}

	public static int getMax(int[] points) {
		int max = -Integer.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (max < points[i])
				max = points[i];
		}
		return max;
	}

	public static int getMin(int[] points) {
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (min > points[i])
				min = points[i];
		}
		return min;
	}

	public static double getArgMin(double[] points) {
		double min = Double.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			if (min > points[i])
				min = points[i];
		}
		return min;
	}

	public static double[] normalizeFrequencies(double[] frequencies, int numberOfPoints) {
		int intervs = frequencies.length;
		for (int i = 0; i < intervs; i++) {
			frequencies[i] = frequencies[i] / (double) numberOfPoints;
		}

		return frequencies;

	}

	// checks if an interval contains at least one element from a sequence of points
	public static boolean intervalContainsPoints(double min, double max, double[] points) {
		// System.out.println(min+"-"+max);
		boolean contains = false;
		for (int i = 0; i < points.length; i++) {
			if ((points[i] >= min) && (points[i] < max)) {
				// System.out.println("---->"+points[i]);
				contains = true;
				break;
			}
		}
		return contains;
	}

	// standardizes a matrix: each row represents a vector: outputs columns means
	// and variances
	double[] means;
	double[] sds;

	public double[][] standardize(double[][] matrix, double[] meansVec, double[] sdVec) {

		int ncols = matrix[0].length;
		int mrows = matrix.length;

		if ((meansVec == null) && (sdVec == null)) {

			meansVec = new double[ncols];
			sdVec = new double[ncols];

			for (int i = 0; i < ncols; i++) {
				meansVec[i] = columnMean(matrix, i);
				sdVec[i] = columnSD(matrix, i);
			}
			means = meansVec;
			sds = sdVec;
		}else {
			means = meansVec;
			sds = sdVec;
		}

		double[][] matrix_std = new double[matrix.length][matrix[0].length];
		for (int i = 0; i < ncols; i++) {
			double mean_i = meansVec[i];
			double sd_i = sdVec[i];
			for (int j = 0; j < mrows; j++) {

				matrix_std[j][i] = (matrix[j][i] - mean_i) / sd_i;

			}
		}

		return matrix_std;
	}

	double[] mins;
	double[] maxs;

	public double[][] normalize01(double[][] matrix, double[] mins, double[] maxs) {

		int ncols = matrix[0].length;
		int mrows = matrix.length;

		if ((mins == null) && (maxs == null)) {

			mins = new double[ncols];
			maxs = new double[ncols];

			for (int i = 0; i < ncols; i++) {
				mins[i] = columnMin(matrix, i);
				maxs[i] = columnMax(matrix, i);
			}
			this.mins = mins;
			this.maxs = maxs;
		}else {
			this.mins = mins;
			this.maxs = maxs;
		}
			

		double[][] matrix_std = new double[matrix.length][matrix[0].length];
		for (int i = 0; i < ncols; i++) {
			double min_i = mins[i];
			double max_i = maxs[i];
			for (int j = 0; j < mrows; j++) {

				matrix_std[j][i] = (matrix[j][i] - min_i) / (max_i - min_i);

			}
		}

		return matrix_std;
	}

	public double[][] normalize01(double[][] matrix) {
		return normalize01(matrix, null, null);
	}

	public double[][] standardize(double[][] matrix) {
		return standardize(matrix, null, null);
	}

	// gets all the columns from a matrix
	public static double[][] traspose(double[][] matrix) {
		int m = matrix.length;
		if (m > 0) {
			int n = matrix[0].length;

			double columns[][] = new double[n][m];

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++)
					columns[i][j] = matrix[j][i];
			}

			return columns;
		} else
			return null;
	}

	// gets a column from a matrix
	public static double[] getColumn(int index, double[][] matrix) {
		int colulen = matrix.length;
		double column[] = new double[colulen];
		for (int i = 0; i < colulen; i++) {
			column[i] = matrix[i][index];
		}
		return column;
	}

	// substitutes a column in a matrix
	public static void substColumn(double[] column, int index, double[][] matrix) {

		for (int i = 0; i < matrix.length; i++) {
			matrix[i][index] = column[i];
		}

	}

	// merge matrixes: puts the rows of a matrix under another matrix
	public static double[][] mergeMatrixes(double[][] matrix1, double[][] matrix2) {

		if ((matrix1 == null) || (matrix1.length == 0))
			return matrix2;
		else if ((matrix2 == null) || (matrix2.length == 0))
			return matrix1;
		else {
			int len1 = matrix1.length;
			int len2 = matrix2.length;
			int superlen = len1 + len2;
			double[][] supermatrix = new double[superlen][];
			for (int i = 0; i < len1; i++) {
				supermatrix[i] = matrix1[i];
			}
			for (int i = len1; i < superlen; i++) {
				supermatrix[i] = matrix2[i - len1];
			}
			return supermatrix;
		}
	}

	public static String vector2String(double[] vector) {
		String out = "";
		for (int i = 0; i < vector.length; i++) {
			if (i > 0)
				out = out + "," + vector[i];
			else
				out = "" + vector[i];
		}

		return out;
	}

	public static double indexString(String string) {
		// string = Sha1.SHA1(string);
		StringBuffer sb = new StringBuffer();
		if ((string == null) || (string.length() == 0))
			return -1;

		int m = string.length();
		for (int i = 0; i < m; i++) {
			sb.append((int) string.charAt(i));
		}

		double d = Double.MAX_VALUE;
		try {
			d = Double.valueOf(sb.toString());
		} catch (Throwable e) {
		}

		if (d > Integer.MAX_VALUE)
			return (indexString(string.substring(0, 3)));

		return d;
	}

	public static double[] initializeVector(int n, double value) {
		double[] numericArray = new double[n];
		for (int i = 0; i < n; i++) {
			numericArray[i] = value;
		}

		return numericArray;
	}

	public static double[] reduceVectorValues(double values[], double reductionFactor) {
		double[] numericArray = new double[values.length];

		for (int i = 0; i < values.length; i++) {
			numericArray[i] = values[i] - reductionFactor * values[i];
		}

		return numericArray;
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

	// ###########Math functions
	public static int powerTwoApproximation(int n) {
		int y = 0;

		while (Math.pow(2, y) <= n) {
			y++;
		}
		y--; // lower approx
		return ((int) Math.pow(2, y));

	}

	public static double dist(double[] d1, double[] d2) {

		double sum = 0;

		for (int i = 0; i < d1.length; i++) {

			sum += (d1[i] - d2[i]) * (d1[i] - d2[i]);

		}

		sum = Math.sqrt(sum);

		return sum;
	}

	public static double distNorm(double[] d1, double[] d2) {

		double sum = 0;
		double dd1 = 0;
		double dd2 = 0;
		for (int i = 0; i < d1.length; i++) {

			sum += (d1[i] - d2[i]) * (d1[i] - d2[i]);
			dd1 += (d1[i]) * (d1[i]);
			dd2 += (d2[i]) * (d2[i]);
		}

		double dist = Math.sqrt(0.5 * sum / (dd1 + dd2));

		return dist;
	}

	public static double angle(double[] d1, double[] d2) {

		double dot = 0;
		double module1 = 0;
		double module2 = 0;

		for (int i = 0; i < d1.length; i++) {

			dot += (d1[i] * d2[i]);
			module1 += (d1[i] * d1[i]);
			module2 += (d2[i] * d2[i]);
		}

		module1 = Math.sqrt(module1);
		module2 = Math.sqrt(module2);

		double angle = Math.acos(dot / (module1 * module2)) * 180 / Math.PI;

		return angle;
	}

	// rounds to the xth decimal position
	public static double roundDecimal(double number, int decimalposition) {

		double n = (double) Math.round(number * Math.pow(10.00, decimalposition)) / Math.pow(10.00, decimalposition);
		return n;
	}

	// increments a percentage o mean calculation when a lot of elements are present
	public static float incrementPerc(float perc, float quantity, int N) {

		if (N == 0)
			return quantity;

		float out = 0;
		int N_plus_1 = N + 1;
		out = (float) ((perc + ((double) quantity / (double) N)) * ((double) N / (double) N_plus_1));
		return out;

	}

	// calculates mean
	public static double mean(double[] p) {
		double sum = 0; // sum of all the elements
		for (int i = 0; i < p.length; i++) {
			sum += p[i];
		}
		return sum / p.length;
	}// end method mean

	// calculates normalized derivative
	public static double[] derivative(double[] a) {
		double[] d = new double[a.length];
		double max = 1;
		if (a.length > 0) {
			for (int i = 0; i < a.length; i++) {
				double current = a[i];
				double previous = current;
				if (i > 0) {
					previous = a[i - 1];
				}
				d[i] = current - previous;
				if (Math.abs(d[i]) > max)
					max = Math.abs(d[i]);
				// System.out.println("point "+a[i]+" derivative "+d[i]);
			}

			// normalize
			for (int i = 0; i < a.length; i++) {
				d[i] = d[i] / max;
			}
		}

		return d;
	}

	// returns a list of spikes indexes
	public static boolean[] findMaxima(double[] derivative, double threshold) {
		boolean[] d = new boolean[derivative.length];

		if (d.length > 0) {
			d[0] = false;
			for (int i = 1; i < derivative.length - 1; i++) {
				if ((derivative[i] / derivative[i + 1] < 0) && derivative[i] > 0) {
//											double ratio = Math.abs((double) derivative[i]/ (double) derivative[i+1]);
//											System.out.println("RATIO "+i+" "+Math.abs(derivative[i]));
//											if ((threshold>0)&&(ratio<threshold))
					if ((threshold > 0) && (Math.abs(derivative[i]) > threshold))
						d[i] = true;
				} else
					d[i] = false;
			}
			double max = getArgMax(derivative);
			if (max == derivative[derivative.length - 1])
				d[derivative.length - 1] = true;
			else
				d[derivative.length - 1] = false;
		}

		return d;
	}

	// returns a list of spikes indexes
	public static boolean[] findSpikes(double[] derivative, double threshold) {
		boolean[] d = new boolean[derivative.length];

		if (d.length > 0) {
			d[0] = false;
			for (int i = 1; i < derivative.length - 1; i++) {
				if (derivative[i] / derivative[i + 1] < 0) {
//										double ratio = Math.abs((double) derivative[i]/ (double) derivative[i+1]);
//										System.out.println("RATIO "+i+" "+Math.abs(derivative[i]));
//										if ((threshold>0)&&(ratio<threshold))
					if ((threshold > 0) && (Math.abs(derivative[i]) > threshold))
						d[i] = true;
				} else
					d[i] = false;
			}
			d[derivative.length - 1] = false;
		}

		return d;
	}

	// returns a list of spikes indexes
	public static boolean[] findSpikes(double[] derivative) {
		return findSpikes(derivative, -1);
	}

	public static BigInteger chunk2Index(int chunkIndex, int chunkSize) {
		return BigInteger.valueOf(chunkIndex).multiply(BigInteger.valueOf(chunkSize));
	}

	public static double[] logSubdivision(double start, double end, int numberOfParts) {

		if (end <= start)
			return null;

		if (start == 0) {
			start = 0.01;
		}
		double logStart = Math.log(start);
		double logEnd = Math.log(end);
		double step = 0;
		if (numberOfParts > 0) {

			double difference = logEnd - logStart;
			step = (difference / (double) numberOfParts);

		}
//						double [] points = new double[numberOfParts+1];
		double[] linearpoints = new double[numberOfParts + 1];

		for (int i = 0; i < numberOfParts + 1; i++) {

//							points[i] = logStart+(i*step);

			linearpoints[i] = Math.exp(logStart + (i * step));
			if (linearpoints[i] < 0.011)
				linearpoints[i] = 0;
		}

		return linearpoints;
	}

	public static double cohensKappaForDichotomy(long NumOf_A1_B1, long NumOf_A1_B0, long NumOf_A0_B1,
			long NumOf_A0_B0) {
		long T = NumOf_A1_B1 + NumOf_A1_B0 + NumOf_A0_B1 + NumOf_A0_B0;

		double Pra = (double) (NumOf_A1_B1 + NumOf_A0_B0) / (double) T;
		double Pre1 = (double) (NumOf_A1_B1 + NumOf_A1_B0) * (double) (NumOf_A1_B1 + NumOf_A0_B1) / (double) (T * T);
		double Pre2 = (double) (NumOf_A0_B0 + NumOf_A0_B1) * (double) (NumOf_A0_B0 + NumOf_A1_B0) / (double) (T * T);
		double Pre = Pre1 + Pre2;
		double Kappa = (Pra - Pre) / (1d - Pre);
		return roundDecimal(Kappa, 3);
	}

	public static String kappaClassificationLandisKoch(double kappa) {
		if (kappa < 0)
			return "Poor";
		else if ((kappa >= 0) && (kappa <= 0.20))
			return "Slight";
		else if ((kappa >= 0.20) && (kappa <= 0.40))
			return "Fair";
		else if ((kappa > 0.40) && (kappa <= 0.60))
			return "Moderate";
		else if ((kappa > 0.60) && (kappa <= 0.80))
			return "Substantial";
		else if (kappa > 0.80)
			return "Almost Perfect";
		else
			return "Not Applicable";
	}

	public static String kappaClassificationFleiss(double kappa) {
		if (kappa < 0)
			return "Poor";
		else if ((kappa >= 0) && (kappa <= 0.40))
			return "Marginal";
		else if ((kappa > 0.4) && (kappa <= 0.75))
			return "Good";
		else if (kappa > 0.75)
			return "Excellent";
		else
			return "Not Applicable";
	}

	public static double scalarProduct(double[] a, double[] b) {

		double sum = 0;

		for (int i = 0; i < a.length; i++) {
			if (i < b.length)
				sum = sum + a[i] * b[i];
		}

		return sum;
	}

	public static double sumVector(double[] a) {

		double sum = 0;

		for (int i = 0; i < a.length; i++) {
			sum = sum + a[i];
		}

		return sum;
	}

	public static double[] vectorialDifference(double[] a, double[] b) {

		double[] diff = new double[a.length];

		for (int i = 0; i < a.length; i++) {
			if (i < b.length)
				diff[i] = a[i] - b[i];
			else
				diff[i] = a[i];
		}

		return diff;
	}

	public static double[] vectorialAbsoluteDifference(double[] a, double[] b) {

		double[] diff = new double[a.length];

		for (int i = 0; i < a.length; i++) {
			if (i < b.length)
				diff[i] = Math.abs(a[i] - b[i]);
			else
				diff[i] = Math.abs(a[i]);
		}

		return diff;
	}

	// calculates the number of elements to take from a set with inverse weight
	// respect to the number of elements
	public static int calcNumOfRepresentativeElements(int numberOfElements, int minimumNumberToTake) {
		return (int) Math.max(minimumNumberToTake, numberOfElements / Math.log10(numberOfElements));
	}

	public static double[] linearInterpolation(double el1, double el2, int intervals) {

		double step = (el2 - el1) / (double) intervals;

		double[] intervalsd = new double[intervals];
		intervalsd[0] = el1;
		for (int i = 1; i < intervals - 1; i++) {
			intervalsd[i] = el1 + step * i;
		}
		intervalsd[intervals - 1] = el2;

		return intervalsd;
	}

	private static double parabol(double a, double b, double c, double x, double shift) {
		return a * (x - shift) * (x - shift) + b * (x - shift) + c;
	}

	public static double[] inverseParabol(double a, double b, double c, double y) {

		double[] ret = { (-1d * b + Math.sqrt(b * b + 4 * a * (Math.abs(y) - c))) / (2 * a),
				(-1d * b - Math.sqrt(b * b + 4 * a * (Math.abs(y) - c))) / (2 * a) };
		return ret;
	}

	public static double logaritmicTransformation(double y) {
		y = Math.abs(y);
		if (y == 0)
			return -Double.MAX_VALUE;
		else
			return Math.log10(y);
	}

	// the parabol is centered on the start Point
	public static double[] parabolicInterpolation(double startP, double endP, int intervals) {

		double start = startP;
		double end = endP;
		double shift = start;

		double a = 1000d;
		double b = 0d;
		double c = 0d;
		double parabolStart = parabol(a, b, c, start, shift);
		if (start < 0)
			parabolStart = -1 * parabolStart;

		double parabolEnd = parabol(a, b, c, end, start);
		if (end < 0)
			parabolEnd = -1 * parabolEnd;

		double step = 0;
		if (intervals > 0) {
			double difference = Math.abs(parabolEnd - parabolStart);
			step = (difference / (double) intervals);
		}

		double[] linearpoints = new double[intervals];

		linearpoints[0] = startP;
		// System.out.println("Y0: "+parabolStart);
		for (int i = 1; i < intervals - 1; i++) {
			double ypoint = 0;
			if (end > start)
				ypoint = parabolStart + (i * step);
			else
				ypoint = parabolStart - (i * step);
			// System.out.println("Y: "+ypoint);
			double res[] = inverseParabol(a, b, c, Math.abs(ypoint));
			// System.out.println("X: "+linearpoints[i]);
			if (ypoint < 0)
				linearpoints[i] = res[1] + shift;
			else
				linearpoints[i] = res[0] + shift;
		}

		linearpoints[intervals - 1] = endP;
		return linearpoints;
	}

	// distributes uniformly elements in parts
	public static int[] takeChunks(int numberOfElements, int partitionFactor) {
		int[] partitions = new int[1];
		if (partitionFactor <= 0) {
			return partitions;
		} else if (partitionFactor == 1) {
			partitions[0] = numberOfElements;
			return partitions;
		}

		int chunksize = numberOfElements / (partitionFactor);
		int rest = numberOfElements % (partitionFactor);
		if (chunksize == 0) {
			partitions = new int[numberOfElements];
			for (int i = 0; i < numberOfElements; i++) {
				partitions[i] = 1;
			}
		} else {
			partitions = new int[partitionFactor];
			for (int i = 0; i < partitionFactor; i++) {
				partitions[i] = chunksize;
			}

			for (int i = 0; i < rest; i++) {
				partitions[i]++;
			}

		}

		return partitions;
	}

	public static int chunkize(int numberOfElements, int partitionFactor) {
		int chunksize = numberOfElements / partitionFactor;
		int rest = numberOfElements % partitionFactor;
		if (chunksize == 0)
			chunksize = 1;
		else if (rest != 0)
			chunksize++;
		/*
		 * int numOfChunks = numberOfElements / chunksize; if ((numberOfElements %
		 * chunksize) != 0) numOfChunks += 1;
		 */

		return chunksize;
	}

	public static double[] uniformSampling(double min, double max, int maxElementsToTake) {
		double step = (max - min) / (double) (maxElementsToTake - 1);
		double[] samples = new double[maxElementsToTake];

		for (int i = 0; i < samples.length; i++) {
			double value = min + i * step;
			if (value > max)
				value = max;
			samples[i] = value;
		}

		return samples;
	}

	public static int[] uniformIntegerSampling(double min, double max, int maxElementsToTake) {
		double step = (max - min) / (double) (maxElementsToTake - 1);
		int[] samples = new int[maxElementsToTake];

		for (int i = 0; i < samples.length; i++) {
			double value = min + i * step;
			if (value > max)
				value = max;
			samples[i] = (int) value;
		}

		return samples;
	}

	// finds the best subdivision for a sequence of numbers
	public static double[] uniformDivide(double max, double min, double[] points) {
		int maxintervals = 10;
		int n = maxintervals;

		boolean subdivisionOK = false;
		double gap = (max - min) / n;

		// search for the best subdivision: find the best n
		while (!subdivisionOK) {
			// System.out.println("*************************");
			boolean notcontains = false;
			// take the gap interval to test
			for (int i = 0; i < n; i++) {
				double rightmost = 0;
				// for the last border take a bit more than max
				if (i == n - 1)
					rightmost = max + 0.01;
				else
					rightmost = min + gap * (i + 1);
				// if the interval doesn't contain any point discard the subdivision
				if (!intervalContainsPoints(min + gap * i, rightmost, points)) {
					notcontains = true;
					break;
				}
			}

			// if there are empty intervals and there is space for another subdivision
			// proceed
			if (notcontains && n > 0) {
				n--;
				gap = (max - min) / n;
			}
			// otherwise take the default subdivision
			else if (n == 0) {
				n = maxintervals;
				subdivisionOK = true;
			}
			// if all the intervals are non empty then exit
			else
				subdivisionOK = true;
		}

		// once the best n is found build the intervals
		double[] intervals = new double[n];
		for (int i = 0; i < n; i++) {
			if (i < n - 1)
				intervals[i] = min + gap * (i + 1);
			else
				intervals[i] = Double.POSITIVE_INFINITY;
		}

		return intervals;
	}

	public static double[] quantiles(double[] data1) {
		double [] data = Arrays.copyOf(data1, data1.length);
		Arrays.sort(data);
		double q1 = calculateQuartile(data, 0.25);
		double q2 = calculateQuartile(data, 0.5);
		double q3 = calculateQuartile(data, 0.75);

		double[] quants = new double[3];
		quants[0] = q1;
		quants[1] = q2;
		quants[2] = q3;
		return quants;
	}

	public static double calculateQuartile(double[] data, double percentile) {
		int n = data.length;
		double index = percentile * (n - 1) + 1;

		if (index % 1 == 0) {
			// If the index is an integer, return the corresponding element
			return data[(int) index - 1];
		} else {
			// If the index is not an integer, interpolate between two adjacent elements
			int lowerIndex = (int) Math.floor(index);
			int upperIndex = (int) Math.ceil(index);

			double lowerValue = data[lowerIndex - 1];
			double upperValue = data[upperIndex - 1];

			return lowerValue + (index - lowerIndex) * (upperValue - lowerValue);
		}
	}

	// calculates the frequency distribution for a set of points respect to a set of
	// intervals
	public static double[] calcFrequencies(double[] interval, double[] points) {
		int intervs = interval.length;
		int npoints = points.length;
		double[] frequencies = new double[intervs];
		for (int i = 0; i < intervs; i++) {

			for (int j = 0; j < npoints; j++) {

				if (((i == 0) && (points[j] < interval[i]))
						|| ((i == intervs - 1) && (points[j] >= interval[i - 1]) && (points[j] <= interval[i]))
						|| ((i > 0) && (points[j] >= interval[i - 1]) && (points[j] < interval[i]))) {
					// System.out.println("(" + (i == 0 ? "" : interval[i - 1]) + "," + interval[i]
					// + ")" + " - " + points[j]);
					frequencies[i] = frequencies[i] + 1;
				}
			}
		}

		return frequencies;
	}

	public static String encodeStrings(String [] input) throws Exception {
		StringBuffer sb = new StringBuffer();
		for (String i:input) {
			sb.append(encodeString(i)+"#");
		}
		
		return compactString(sb.toString());
	}
	
	public static String compactString(String input) throws Exception {
		int len = input.length();
		
		LinkedHashMap<String, Integer> m = new LinkedHashMap<String, Integer>();
		for (int i=0;i<len;i++) {
			String c = ""+input.charAt(i);
			if (m.get(c)==null)
				m.put(c, 1);
			else
				m.put(c, m.get(c)+1);
		}
		
		StringBuffer sb = new StringBuffer();
		for (String mc : m.keySet()) {
			sb.append(mc+""+m.get(mc));
		}
		return sb.toString();
	}
	
	public static String encodeString(String input) throws Exception {
		try {
			// Create a SHA-256 message digest instance
			MessageDigest digest = MessageDigest.getInstance("SHA-256");

			// Encode the input string into a byte array
			byte[] encodedHash = digest.digest(input.getBytes(StandardCharsets.UTF_8));

			// Convert the byte array to a hexadecimal string representation
			return bytesToHex(encodedHash);

		} catch (Exception e) {
			throw new RuntimeException("Error: SHA-256 algorithm not found.");
		}
	}

	

//Helper method to convert a byte array to a hexadecimal string
	private static String bytesToHex(byte[] bytes) {
		StringBuilder hexString = new StringBuilder(2 * bytes.length);
		for (byte b : bytes) {
			String hex = Integer.toHexString(0xff & b);
			if (hex.length() == 1) {
				hexString.append('0');
			}
			hexString.append(hex);
		}
		return hexString.toString();
	}

}
