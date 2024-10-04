package it.cnr.anomaly;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class TestSerialization {


	    public static void main(String[] args) {
	        AnomalyDetectionVariationalAutoEncoder vae = new AnomalyDetectionVariationalAutoEncoder();

	        // Serialization
	        try (FileOutputStream fileOut = new FileOutputStream("vae_test.obj");
	             ObjectOutputStream objectOut = new ObjectOutputStream(fileOut)) {

	            objectOut.writeObject(vae);
	            System.out.println("The VAE was successfully written to vae_test.obj");

	        } catch (IOException e) {
	            e.printStackTrace();
	        }

	        // Deserialization
	        try (FileInputStream fileIn = new FileInputStream("vae_test.obj");
	             ObjectInputStream objectIn = new ObjectInputStream(fileIn)) {

	            AnomalyDetectionVariationalAutoEncoder mn = (AnomalyDetectionVariationalAutoEncoder) objectIn.readObject();
	            System.out.println("The VAE was successfully read from vae_test.obj");

	        } catch (IOException | ClassNotFoundException e) {
	            e.printStackTrace();
	        }
	    }

}
