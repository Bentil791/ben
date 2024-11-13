import java.util.Scanner;
class MyException extends Exception
{
	private double value;
	
	MyException(double a)
	{
		value = a;
	}
	
	MyException(String message, double a)
	{	
		super(message);
		value=a;
	}

	public double getValue(){return value;}
	
	public String toString()
	{
		return "VALUE : " + value;
	}
}


class Point
{
		private double x,y;
		public Point(double _x, double _y)
		{
			x=_x;
			y=_y;
		}
		
		public double dist()
		{
			return Math.sqrt(x*x+y*y);
		}
		
		public double dist(Point p2) throws MyException
		{
			double r=Math.sqrt(Math.pow((x-p2.x),2)+Math.pow((y-p2.y),2));
			if (r>50) throw new MyException("the distance is too big",r);
			else return r;
		}
		
		@Override
		public String toString()
		{
			return "( " + x + " ; " + y + " )";
		}
		
public static void main(String args[]) 
	{
		
		Scanner in = new Scanner(System.in);
        System.out.print("Input the X coord for point p1 : ");
        int p1x = in.nextInt();
        
        System.out.print("Input the Y coord for point p1 : ");
        int p1y = in.nextInt();
        
        System.out.print("Input the X coord for point p2 : ");
        int p2x = in.nextInt();
        
        System.out.print("Input the Y coord for point p2 : ");
        int p2y = in.nextInt();
        
		Point p1 = new Point(p1x,p1y);
		Point p2 = new Point(p2x,p2y);
		System.out.println("p1 "+p1);
		System.out.println("p2 "+p2);
		System.out.println("p1 dist to (0;0) "+p1.dist());
		try{ System.out.println("p1 dist to p2 "+p1.dist(p2)); }
		catch(MyException e){System.out.println(e);
		System.out.println(e.getMessage()+" "+ e.getValue());
		}
		
	};
	
}
