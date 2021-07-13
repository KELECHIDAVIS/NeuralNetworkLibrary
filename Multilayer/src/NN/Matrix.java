package NN;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class Matrix implements Serializable
{
	
	public int rows, cols; 
	public float[][] data; 
	public Matrix(int rows,int cols)
	{
		this.rows= rows ; 
		this.cols = cols; 
		this.data = new float[rows][cols]; 
		for(int i =0; i<rows;i++)
		{
			for(int y = 0; y<cols; y++)
			{
				this.data[i][y] = 0; 
				
			}
		}
		
		
	}
	public Matrix copy()
	{
		Matrix m = new Matrix(this.rows,this.cols); 
		for(int i = 0; i<this.rows; i++)
		{
			for(int j=0; j<this.cols;j++)
			{
				m.data[i][j] = this.data[i][j]; 
			}
		}
		return m; 
		
	}
	public void sigmoid() { // the sigmoid activation function 
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
                this.data[i][j] = (float) (1/(1+Math.exp(-this.data[i][j]))); 
        }
        
    }
	public void dsigmoid() { //returns a matrix ; the derivative of the sigmoid ; this is required when calculating the gradients for backpropogation 
        //if something doesn't work maybe have this happen the the actual data
        //its going to assume it has already been sigmoided 
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
                this.data[i][j] = this.data[i][j] * (1-this.data[i][j]);
        }
        
    }
	
	//static version of dsigmoid 
	public static Matrix dsigmoid(Matrix m) { //returns a matrix ; the derivative of the sigmoid ; this is required when calculating the gradients for backpropogation 
        Matrix temp=new Matrix(m.rows,m.cols);
        for(int i=0;i<m.rows;i++)
        {
            for(int j=0;j<m.cols;j++)
                temp.data[i][j] = m.data[i][j] * (1-m.data[i][j]);
        }
        return temp;
        
        
        
    }
	
	
	public void show()
	{
		for(int i =0; i<rows;i++)
		{
			for(int y = 0; y<cols; y++)
			{
				System.out.print(this.data[i][y]+" " );
				
			}
			System.out.println();
		}
	}
	public static Matrix transpose(Matrix m)
	{
		Matrix result = new Matrix(m.cols,m.rows);
		
		for(int i =0; i<m.rows;i++)
		{
			for(int y = 0; y<m.cols; y++)
			{
				result.data[y][i]= m.data[i][y];   
				
			}
		}
		
		return result; 
	}
	public void randomize()
	{
		Random rand = new Random(); 
		for(int i =0; i<rows;i++)
		{
			for(int y = 0; y<cols; y++)
			{
				this.data[i][y]=rand.nextFloat()*2-1;  
				
			}
		}
	}
	
	// scalar functions 
	public void mult(double n )
	{
		for(int i =0; i<rows;i++)
		{
			for(int y = 0; y<cols; y++)
			{
				this.data[i][y]*=n;  
				
			}
		}
	}
	public void add(double n )
	{
		for(int i =0; i<rows;i++)
		{
			for(int y = 0; y<cols; y++)
			{
				this.data[i][y]+=n;  
				
			}
		}
	}
	
	// elements wise ; matrixes have to be the same size otherwise error 
	public void add(Matrix m)
	{
		for(int i =0; i<rows;i++)
		{
			for(int y = 0; y<cols; y++)
			{
				this.data[i][y]+=m.data[i][y];  
				
			}
		}
	}
	public void mult(Matrix m)// this is for elementwise 
	{
		
		for(int i =0; i<rows;i++)
		{
			for(int y = 0; y<cols; y++)
			{
				this.data[i][y]*=m.data[i][y];  
				
			}
		}
			
		
	}
	 // these next two functions are helper functions to help convert a matrix to 
    public static Matrix fromArray(float[] x)
    {
        Matrix temp = new Matrix(x.length,1);
        for(int i =0;i<x.length;i++)
            temp.data[i][0]=x[i];
        
        return temp;
        
    }
	
    public float[] toArray() {
        float[] temp = new float[rows*cols]; 
        ArrayList<Float> fake = new ArrayList<Float>(); 
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
            	fake.add(data[i][j]);
            }
        }
        
        for(int i =0; i<fake.size(); i++)
        {
        	temp[i] = fake.get(i); 
        	
        }
        return temp;
   }
    
    public static Matrix subtract(Matrix a, Matrix b) {
    	//if error occurs it was probably because of the dimensions when subtracting
    	
        Matrix temp=new Matrix(a.rows,a.cols);
        for(int i=0;i<a.rows;i++)
        {
            for(int j=0;j<a.cols;j++)
            {
                temp.data[i][j]=a.data[i][j]-b.data[i][j];// this 
            }
        }
        return temp;
    }
    
	public static Matrix dot(Matrix a, Matrix b) { // this is a static method which could be used to multiply two matrixes ;  dot product 
        Matrix temp=new Matrix(a.rows,b.cols);
        for(int i=0;i<temp.rows;i++)
        {
            for(int j=0;j<temp.cols;j++)
            {
                float sum=0;
                for(int k=0;k<a.cols;k++)
                {
                    sum+=a.data[i][k]*b.data[k][j];
                }
                temp.data[i][j]=sum;
            }
        }
        return temp;
    }

}
