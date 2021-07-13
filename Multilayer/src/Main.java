import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import NN.NeuralNetwork;

public class Main
{
	
	static class Training
	{
		float[] input, targets; 
		
		public Training(float[] inputs, float[] targets)
		{
			this.input = inputs ; 
			this.targets=targets; 
			
		}
		
	}

	public static void main(String[] args)
	{
		
		
		try
		{
			
			NeuralNetwork n =NeuralNetwork.load("res/last.txt"); 
			System.out.println(n.input);
			
			
			
		} catch(Exception e )
		{
			e.printStackTrace(); 
		}

		
	}

}
