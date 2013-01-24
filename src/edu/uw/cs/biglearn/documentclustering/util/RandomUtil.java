package edu.uw.cs.biglearn.documentclustering.util;

import Jama.Matrix;

public class RandomUtil {
	public static double dnorm(double[] x, double[] mean, Matrix cov, boolean isInv) {
		int d = x.length;
		Matrix vec = (new Matrix(x, d)).minus(new Matrix(mean, d));
		double ret = 0.0;
		if (isInv) {
			double y = -0.5 * vec.transpose().times(cov).times(vec).get(0, 0);
			ret = Math.pow(2*Math.PI, -0.5 * d) * Math.pow(cov.det(), -0.5) * Math.exp(y);
		} else {
			double y = -0.5 * vec.transpose().times(cov.inverse()).times(vec).get(0, 0);
			ret = Math.pow(2*Math.PI, -0.5 * d) * Math.pow(cov.det(), -0.5) * Math.exp(y);
		}
		return Double.isNaN(ret) ? 0 : ret;
	}
	
	public static double dnorm(Matrix x, Matrix mean, Matrix cov, boolean isInv) {
		int d = x.getRowDimension();
		Matrix vec = x.minus(mean);
		double ret = 0.0;
		if (isInv) {
			double y = -0.5 * vec.transpose().times(cov).times(vec).get(0, 0);
			ret = Math.pow(2*Math.PI, -0.5 * d) * Math.pow(cov.det(), -0.5) * Math.exp(y);
		} else {
			double y = -0.5 * vec.transpose().times(cov.inverse()).times(vec).get(0, 0);
			ret = Math.pow(2*Math.PI, -0.5 * d) * Math.pow(cov.det(), -0.5) * Math.exp(y);
		}
		return Double.isNaN(ret) ? 0 : ret;
	}
	
	public static void main(String args[]) {
		int p = 100;
		for (int i = 0; i < 10000; i++) {
			Matrix x = Matrix.random(p, 1);
			Matrix mu = Matrix.random(p,1);
			Matrix cov = Matrix.identity(p,p);
			dnorm(x, mu, cov, true);
		}
		System.out.println("done");
	}
}
