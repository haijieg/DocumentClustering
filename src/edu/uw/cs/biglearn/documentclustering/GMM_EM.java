package edu.uw.cs.biglearn.documentclustering;

import java.util.ArrayList;
import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

public class GMM_EM {
Dataset dataset;
	
	public GMM_EM(Dataset dataset) {
		this.dataset = dataset;
	}
	
	public class Cluster {
		double[] mean;
		SimpleMatrix cov;
		String id;
		public Cluster(String id, double[] mean) {
			this.mean = mean.clone();
			int p = mean.length;
			cov = SimpleMatrix.identity(p);
		}
	}
	
	public void runEM(ArrayList<Cluster> clusters, int maxIter) {
		int n = dataset.sampleSize(); int k = clusters.size();
		double[][] rik = new double[n][k];
		Arrays.fill(rik, 1.0/k);
		
		int iter = 0;
		double ll = 0.0;
		while (iter < maxIter) {
			EStep();
			MStep();
			
			double newll = loglikelihood();
			System.out.println("loglikelihood: " + newll);
			
			if (Math.abs(newll-ll) < 1e-4)
				break;			
			ll = newll;
			iter++;
		}
		
	}
	
	private double loglikelihood() {
		return 0.0;
	}
	
	private void EStep() {
		
	}
	
	private void MStep() {
		
	}
}
