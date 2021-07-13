package NN;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;

public class NeuralNetwork implements Serializable 
{
	
	//you would want to normalize all the data before passing it through the neural network 
	// we are going to store the weights as a matrix
	public int input ,hidden,output; 
	public float learningRate = 0.1f; 
	public Matrix weightIH,weightHO, biasH, biasO; //weights between the input -hidden then to the hidden-output; hidden and output bias matrixes
	
	public NeuralNetwork(NeuralNetwork nn)
	{
		this.input = nn.input; 
		this.output=nn.output; 
		this.hidden = nn.hidden; 
		this.weightIH =nn.weightIH.copy(); 
		this.weightHO = nn.weightHO.copy(); 
		this.biasH = nn.biasH.copy(); 
		this.biasO = nn.biasO.copy(); 
		
	}
	public NeuralNetwork(int input, int hidden, int output) // it takes in the amount of neurons per layer
	{
		
		this.input= input; 
		this.hidden = hidden ; 
		this.output=output; 
		
		this.weightIH = new Matrix(this.hidden,this.input); 
		this.weightHO = new Matrix(this.output,this.hidden); 
		
		this.weightIH.randomize();
		this.weightHO.randomize();
		
		this.biasH = new Matrix(this.hidden,1); 
		this.biasO = new Matrix(this.output,1); 
		
		this.biasH.randomize();
		this.biasO.randomize();
		
		
	}

	public float[] predict(float[] inputArray) // weighed sum;  takes the input multiplied by the weight by each input then adds them all together with one bias, then it passes that sum to another neuron as an input 
	{ 									// after you get the weighted sum you would want to pass it through an activation function  
		//lot of matrix math 
		//GENERATE HIDDEN OUTPUTS
		// turns the input array into a matrix
		Matrix in = Matrix.fromArray(inputArray); 
		
		//this multplies the inputs with the weights and then adds the bias
		Matrix hidden = Matrix.dot(this.weightIH, in ); 
		hidden.add(this.biasH);
		
		//this is the activation function
		hidden.sigmoid();
		
		
		//generating the output's output 
		Matrix output = Matrix.dot(this.weightHO, hidden); 
		output.add(this.biasO);
		
		output.sigmoid();
		
		
		
		
		
		//this is going to return a guess 
		return output.toArray(); 
		
	}
	
	public void train(float[] inputArray, float[] target) // to train you feed this inputs and a known output ; backpropogation ; this is basically feeding the error as output throughout the network backwars 
	{
		
			Matrix in = Matrix.fromArray(inputArray); 
			Matrix hidden = Matrix.dot(this.weightIH, in ); 
			hidden.add(this.biasH);
			
			//this is the activation function
			hidden.sigmoid();
			
			
			//generating the output's output 
			Matrix outputs = Matrix.dot(this.weightHO, hidden); 
			outputs.add(this.biasO);
			
			outputs.sigmoid();
				
	
			//Convert to matrixes 
			
			Matrix targets = Matrix.fromArray(target); 
			
			
			//Calculate error 
			//Error = targets-outputs
			Matrix outputErrors = Matrix.subtract(targets, outputs); 
			
			
			//calculate gradient
			Matrix gradients = Matrix.dsigmoid(outputs);//this dsigmoid could be a problem but right now im going to make it the same a sigmoid
			gradients.mult(outputErrors);
			gradients.mult(this.learningRate); 
			
			
			
		
			//Calculate deltas 
			Matrix hiddenT = Matrix.transpose(hidden); 
			Matrix weightHODelta = Matrix.dot(gradients , hiddenT);
			
			
			//adjusting weights by deltas
			this.weightHO.add(weightHODelta);
			//adjusting the bias by its deltas (which is just the gradients)
			this.biasO.add(gradients);
			
			
			
			//Calculate the hidden layer errors
			Matrix whoT = Matrix.transpose(this.weightHO); 
			Matrix hiddenErrors = Matrix.dot(whoT, outputErrors); 
			
			
			/// now do the same for the ih chunk
			//calculate hidden gradient 
			Matrix hiddenG = Matrix.dsigmoid(hidden); 
			hiddenG.mult(hiddenErrors);
			hiddenG.mult(this.learningRate);
			
			// Calculate input to hidden deltas 
			Matrix inputsT = Matrix.transpose(in	); 
			Matrix weightIHDelta = Matrix.dot(hiddenG, inputsT); 
			
			this.weightIH.add(weightIHDelta);
			//adjusting the bias by its deltas (which is just the gradients)
			this.biasH.add(hiddenG);
			
			
		
		
	}
	
	public void save(String filename) throws FileNotFoundException, IOException
	{
		File f = new File(filename);
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(f)); 
		
		out.writeObject(this); 
		out.flush();
		out.close();
		
	}
	public static NeuralNetwork load(String filename) throws FileNotFoundException, IOException, ClassNotFoundException
	{
		File f = new File(filename);
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(f)); 
		NeuralNetwork n = (NeuralNetwork) in.readObject();
		in.close(); 
		return n; 
		
	}
	
	public NeuralNetwork copy()
	{
		return new  NeuralNetwork(this); 
	}
	public void mutate(float rate, float deviation) // you want to do the mutation to every single thing in each matrix
	{
		Random rand = new Random(); 
		// for each of these arrays mutate its element by the mutation rate; 
		for(int i =0 ; i <this.weightIH.rows; i++)
		{
			for(int y=0 ; y <this.weightIH.cols; y++)
			{
				if((rand.nextFloat()*2-1)<rate)
				{
					this.weightIH.data[i][y]+=rand.nextGaussian()*deviation; 
				}
			}
		}
		for(int i =0 ; i <this.weightHO.rows; i++)
		{
			for(int y=0 ; y <this.weightHO.cols; y++)
			{
				if((rand.nextFloat()*2-1)<rate)
				{
					this.weightHO.data[i][y]+=rand.nextGaussian()*deviation; 
				}
			}
		}
		for(int i =0 ; i <this.biasH.rows; i++)
		{
			for(int y=0 ; y <this.biasH.cols; y++)
			{
				if((rand.nextFloat()*2-1)<rate)
				{
					this.biasH.data[i][y]+=rand.nextGaussian()*deviation; 
				}
			}
		}
		for(int i =0 ; i <this.biasO.rows; i++)
		{
			for(int y=0 ; y <this.biasO.cols; y++)
			{
				if((rand.nextFloat()*2-1)<rate)
				{
					this.biasO.data[i][y]+=rand.nextGaussian()*deviation; 
				}
			}
		}
		
		
		
	
		
		
		
	}

}
