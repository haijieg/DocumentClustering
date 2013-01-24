package edu.uw.cs.biglearn.documentclustering;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import Jama.Matrix;
import edu.uw.cs.biglearn.documentclustering.util.RandomUtil;


public class GMM_EM {
	Dataset dataset;
	
	public GMM_EM(Dataset dataset) {
		this.dataset = dataset;
	}
	
	public class Cluster {
		Matrix mu;
		Matrix sigma;
		Matrix omega;
		String id;
		public Cluster(String id, double[] mean) {
			this.mu = new Matrix(mean, mean.length);
			int p = mean.length;
			sigma = Matrix.identity(p,p);
			omega = sigma.inverse();
		}
	}
	
	public void runEM(ArrayList<Cluster> clusters, int maxIter) {
		int n = dataset.sampleSize(); int k = clusters.size();
		double[][] rik = new double[n][k];
		double[] pi = new double[k];
		for (int i = 0;i < n;i++)
			Arrays.fill(rik[i], 1.0/k);
		Arrays.fill(pi, 1.0/k);
		int iter = 0;
		double ll = 0.0;
		while (iter < maxIter) {
			EStep(pi, rik, clusters);
			MStep(pi, rik, clusters);
			
			double newll = loglikelihood(pi, clusters);
			System.out.println("loglikelihood: " + newll);
			
			if (Math.abs(newll-ll) < 1e-4)
				break;			
			ll = newll;
			iter++;
		}
	}
	
	private double loglikelihood(double[] pk, List<Cluster> clusters) {
		double ret = 0.0;
		int p = dataset.featureDim();
		for (DataInstance di : dataset) {
			for (int i = 0; i < pk.length; i++)
				ret += pk[i] * RandomUtil.dnorm(new Matrix(di.features, p), clusters.get(i).mu, clusters.get(i).omega, true);
		}
		return Math.log(ret);
	}
	
	private void EStep(double[] pi, double[][] rik, List<Cluster> clusters) {
		int p = dataset.featureDim();
		for (int i = 0; i < dataset.sampleSize(); i++) {
			double sum = 0.0;
			for (int k = 0; k < clusters.size(); k++) {
				rik[i][k] = pi[k] * RandomUtil.dnorm(new Matrix(dataset.get(i).features, p), clusters.get(k).mu, clusters.get(k).omega, true);
				sum += rik[i][k];
			}
			for (int k = 0; k < clusters.size(); k++) {
				rik[i][k] /= sum;
			}
		}
	}
	
	private void MStep(double[] pi, double[][]rik, List<Cluster> clusters) {
		int n = dataset.sampleSize();
		int p = dataset.featureDim();
		// update pi
		Arrays.fill(pi, 0);
		for (int i = 0; i < n; i++) {
			for (int k = 0; k < pi.length; k++) {
				pi[k] += rik[i][k];
			}
		}
		for (int k = 0; k < pi.length; k++)
			pi[k] /= n;
		
		// update mu
		for (int k = 0; k < clusters.size(); k++) {
			Matrix mu = new Matrix(p, 1, 0);
			for (int i = 0; i < n; i++) {
				mu.plusEquals( (new Matrix(dataset.get(i).features, p)).times(rik[i][k]) );
			}
			mu.timesEquals(1/(pi[k] * n));
			clusters.get(k).mu = mu;
		}
		
		// update cov
		for (int k = 0; k < clusters.size(); k++) {
			Matrix sigma = new Matrix(p,p,0);
			for (int i = 0; i < n; i++) {
				Matrix x = new Matrix(dataset.get(i).features, p);
				sigma.plusEquals( x.times(x.transpose()).times(rik[i][k]));
			}
			sigma.timesEquals(1/(pi[k]*n));
			Matrix mu = clusters.get(k).mu;
			sigma.minusEquals(mu.times(mu.transpose()));
			clusters.get(k).sigma = sigma;
			clusters.get(k).omega = sigma.inverse();
		}
	}
	
		
	public static void main(String[] args) throws FileNotFoundException, IOException {
		Dataset dataset = new Dataset("data/");
		GMM_EM gmm = new GMM_EM(dataset);
		
		ArrayList<Cluster> clusters = new ArrayList<Cluster>();
		clusters.add(gmm.new Cluster("Cluster_0", dataset.get(20).features));
		clusters.add(gmm.new Cluster("Cluster_1", dataset.get(260).features));
		gmm.runEM(clusters, 10);
	}
}
