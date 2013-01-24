package edu.uw.cs.biglearn.documentclustering;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;

public class Dataset implements Iterable<DataInstance>{
	
	private ArrayList<String> dictionary;
	private ArrayList<DataInstance> rows;
	
	public Dataset (String path) throws FileNotFoundException, IOException {
		dictionary = new ArrayList<String>();
		rows = new ArrayList<DataInstance>();
		loadDictionary(path + "dictionary.txt");
		loadData(path + "tfidf.txt");
	}
	
	public int featureDim () {
		return dictionary.size();
	}
	
	public int sampleSize() {
		return rows.size();
	}
	
	public DataInstance get(int i) {
		return rows.get(i);
	}
	
	public String getWord(int i) {
		return dictionary.get(i);
	}
	
	
	private void loadDictionary(String path) throws FileNotFoundException {
		Scanner sc = new Scanner(new BufferedReader(new FileReader(path)));
		while(sc.hasNextLine()) {
			String line = sc.nextLine();
			dictionary.add(line.split(" ")[0]);
		}
	}
	
	private void loadData(String path) throws FileNotFoundException {
		Scanner sc = new Scanner(new BufferedReader(new FileReader(path)));
		while(sc.hasNextLine()) {
			String line = sc.nextLine();
			rows.add(new DataInstance(line));
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		Dataset dataset = new Dataset("data/");
		System.out.println("N = " + dataset.sampleSize());
		System.out.println("P = " + dataset.featureDim());
		for (int i = 0; i < 100; i++)
			System.out.println(dataset.rows.get(i));
	}

	@Override
	public Iterator<DataInstance> iterator() {
		return rows.iterator();
	}
}
