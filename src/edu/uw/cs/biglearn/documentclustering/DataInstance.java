package edu.uw.cs.biglearn.documentclustering;

import edu.uw.cs.biglearn.documentclustering.util.StringUtil;

public class DataInstance {
	static final int dim = 100; 
	String id;
	double[] features;
	
	public DataInstance(String id, double[] features) {
		this.id = id;
		this.features = features.clone();
	}
	
	public DataInstance(String line) {
		String[] splits = line.split("\\|");
		id = splits[0];
		features = new double[dim];
		String[] tokens = (splits[1].split(","));
		for (String token: tokens) {
			String[] pairs = token.split(":");
			features[Integer.parseInt(pairs[0])] = Double.parseDouble(pairs[1]);
		}
	}
	
	public DataInstance(DataInstance other) {
		this.id = other.id;
		this.features = other.features.clone();
	}
	
	public double distl2(DataInstance other) {
		double dist = 0.0;
		for (int i = 0; i < dim; i++)
			dist += Math.pow((features[i]-other.features[i]),2);
		return Math.sqrt(dist);
	}
	
	@Override
	public String toString() {
		return id + ": " + StringUtil.implode(features, ",");
	}
}
