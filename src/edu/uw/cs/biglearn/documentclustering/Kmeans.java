package edu.uw.cs.biglearn.documentclustering;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;

public class Kmeans {
	Dataset dataset;
	
	public Kmeans(Dataset dataset) {
		this.dataset = dataset;
	}
	
	
	public class Cluster {
		Set<Integer> members;
		DataInstance center;
		Cluster(String id, double[] features) {
			center = new DataInstance(id, features);
			members = new HashSet<Integer>();
		}
		
		@Override
		public String toString() {
			StringBuilder memberstr = new StringBuilder(center.id + ": ");
			ArrayList<Integer> articleids = new ArrayList<Integer>(members);
			Collections.sort(articleids, new Comparator<Integer>() {
				@Override
				public int compare(Integer o1, Integer o2) {
					if (dataset.get(o1).distl2(center) < dataset.get(o2).distl2(center))
						return 1;
					else
						return -1;
				}
			});
			for (int i : articleids) {
				memberstr.append(dataset.get(i).id + " ");
			}
			
			ArrayList<Integer> indices = new ArrayList<Integer>(DataInstance.dim);
			for (int i = 0; i < DataInstance.dim; i++)
				indices.add(i);			
			Collections.sort(indices, new Comparator<Integer>() {
				@Override
				public int compare(Integer arg0, Integer arg1) {
					if (center.features[arg0] > center.features[arg1]) {
						return -1;
					} else {
						return 1;
					}
				}
			});			
			StringBuilder words = new StringBuilder("Words: ");
			for (int i : indices) {
				if (center.features[i] < 0.1) break;
				words.append(dataset.getWord(indices.get(i)) + " ");
			}
			
			return memberstr + "\n" + words;
		}
	}
	
	public ArrayList<Cluster> computeKmeans(ArrayList<DataInstance> seeds, int maxIter) {
		int iter = 0;
		int k = seeds.size();
		int n = dataset.sampleSize();
		
		// initialize assignment and clusters
		int[] assignments = new int[n]; Arrays.fill(assignments, -1);
		ArrayList<Cluster> clusters = new ArrayList<Cluster>(k);
		for (int i = 0; i < k; i++)
			clusters.add(new Cluster("Cluster_"+i, seeds.get(i).features));
		
		while(iter < maxIter) {
			// compute assignment
			for (int i = 0; i < n; i++) {
				int label = computeCluster(dataset.get(i), clusters);
				if (label != assignments[i]) {
					 clusters.get(label).members.add(i);
				if (assignments[i] != -1)
					 clusters.get(label).members.remove(i);
				assignments[i] = label;
				}
			}
			
			// update center
			double delta = 0.0;
			for (Cluster c : clusters) {
				double[] newcenter = new double[DataInstance.dim];
				for (int i : c.members) {
					double[] featurei = dataset.get(i).features;
					for (int j = 0; j < DataInstance.dim; j++)
						newcenter[j] += featurei[j];
				}
				double deltai = 0.0;
				
				for (int j = 0; j < DataInstance.dim; j++) {
					newcenter[j] /= c.members.size();
					deltai += Math.pow((newcenter[j] - c.center.features[j]),2);
					deltai = Math.sqrt(deltai);
				}
				
				c.center.features = newcenter;
				System.out.println(c.center.id + " shifted " + deltai);
				delta += deltai;
			}
			System.out.println("Total shifted " + delta);
			if (Math.abs(delta) < 1e-5)
				break;			
			iter++;
		}
		
		return clusters;
	}
	
	private int computeCluster(DataInstance instance, ArrayList<Cluster> clusters) {
		int label = -1; double min = Double.MAX_VALUE;
		for (int i = 0; i < clusters.size(); i++) {
			DataInstance center = clusters.get(i).center;
			double dist = center.distl2(instance);
			if (dist < min) {
				min = dist;
				label = i;
			}
		}
		return label;
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		Dataset dataset = new Dataset("data/");
		Kmeans kmeans = new Kmeans(dataset);
		ArrayList<DataInstance> seeds = new ArrayList<DataInstance>();
		seeds.add(dataset.get(20)); // u2
		seeds.add(dataset.get(260)); // feminismjapan
		seeds.add(dataset.get(321)); // foreignpolicy 
		seeds.add(dataset.get(44)); // benningtoncollege
		seeds.add(dataset.get(46)); // raceandcrime
		seeds.add(dataset.get(52)); // historyofmexico
		seeds.add(dataset.get(206)); // bryantbulldog
		seeds.add(dataset.get(294)); // history of mathematics

		
		
		ArrayList<Cluster> clusters = kmeans.computeKmeans(seeds, 10);
		for (Cluster cluster : clusters)
			System.out.println(cluster + "\n");
	}
}
