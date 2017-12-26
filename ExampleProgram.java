class ExampleProgram {

	public static void main(String[] args){
	/*	SparseMatrix A = new SparseMatrix(10);
		A.wipeIdentity();
		A.put(0,1,.1f);A.put(1,2,.1f);A.put(2,3,.1f);
		A.print(3);
		A.remove(1,1); A.remove(3,3); A.put(4,4,0f);
		A = A.Transpose();
		A.print(3);
		
		SparseMatrix B = new SparseMatrix(10);
		B.wipeIdentity();
		A.plus(B);
		A.plus(3,5,1.0f);
		A.plus(0,0,.1f);
		A.print(3);
		
		SparseMatrix C = new SparseMatrix(4);
		C.put(0,1,.1f);C.put(0,2,.1f);C.put(3,0,.4f);
		C.put(1,2,.3f);C.put(1,3,.2f);C.put(2,3,.9f);
		
		transferData X = C.TransPr(10);
		//X.print(A.log10(A.getNumRows()-1));
		C.print(1);
		X.kProbs[0].print(3);
		X.kProbs[1].print(3);
		X.kProbs[2].print(3);
		X.kProbs[3].print(3);
		X.kProbs[6].print(3);
		X.kProbs[9].print(3);
		System.out.println('\n');
		int[] Walk = {3,0,1,3,0};
		System.out.println(C.WalkPr(Walk));
		System.out.println(C.shortestLoop());
		System.out.println(C.numNZ());
		System.out.println(C.getNumRows());
		C.print(1);
		C.square().print(3);
		C.multiplyTranspose(C).print(3);
		C.AAt().print(3);
		C.AtA().print(3);
		C.SimRankRegular(6,0.9f).print(10);
		System.out.println(1.9f/1000);
		System.out.println(C.Sampling(3, 2, 6, 1000000, 0.9f));*/
		SparseMatrix D = new SparseMatrix(5);
		D.put(0,1,1.0f);D.put(1,2,1.0f);D.put(2,0,.5f);
		D.put(3,0,1.0f);
		D.SimRankRegular(11,0.6f).print(5);
		/*System.out.println("AAAA");
		D.SimRankRegular(10,0.9f).print(10);
		System.out.println("BBBB");
		D.SimRankRegular(3,0.9f).print(10);
		System.out.println("CCCC");

		SparseMatrix E = new SparseMatrix(4);
		SparseMatrix F = new SparseMatrix(4);
		for (int i=0;i<4;i++){
			for (int j=0;j<4;j++){
				E.put(i,j,D.Sampling(i,j,11,1000,0.9f));
				F.put(i,j,   D.Mixed(i,j,11,1000,0.9f,5));

			}
		}
		E.print(10);
		F.print(10);

		//System.out.println(D.Sampling(0,3,6,1000000,0.9f));
		
		*/
		
		/*for (WalkNode W : X.Walks){
			WalkNode E = W;
			while (E!=null){
				for (int o : E.Walk){
					System.out.print(o);
				}
				System.out.print('\n');
				E = E.next;
			}
		}*/
		
	}
}