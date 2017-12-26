// Implementation of a SparseMatrix as an array of hashtable
// in order to test the algorithms outlined in https://arxiv.org/pdf/1512.02714.pdf
import java.util.Random;
import java.util.Arrays;



class SparseMatrix{
	// This is a sparse Matrix implemented as a array of hashtables
	// each hashtable represents a row of our matrix (A)
	
	private final int numRows; // size of (rows)
	private final int sizeHash; // size of the HashTable's stored in (rows)

	private HashTable[] rows; // array of HashTable's storing our 2d sparse Matrix
	
	public SparseMatrix (int n){
		numRows = n;
		sizeHash = n/16384+10;
		rows = new HashTable[n];
	}
	
	public void put(int i, int j, float value){ // puts (value) into position (i,j)
		if (rows[i]==null) {
			rows[i] = new HashTable(sizeHash);
		}
		rows[i].put(j,value);
	}
	
	public void remove(int i, int j){
		if (rows[i]!=null){
			rows[i].remove(j);
		}
	}
	
	public float get(int i, int j){ // returns (i,j) entry of A
		if (rows[i]==null) {
			return 0.0f;
		}
		return rows[i].get(j);
	}
	
	public int numNZ(){ // returns the number of non-zero entries in our sparse matrix
		int sum = 0;
		for (int i =0; i<numRows;i++){
			if (rows[i]!=null){
				sum += rows[i].numEntries;
			}
		}
		return sum;
	}
	
	public int getNumRows(){
		return numRows;
	}
	
	public void wipeIdentity() { // converts the table into Identity matrix
		this.rows = new HashTable[numRows];
		for (int i =0; i<numRows; i++){
			put(i,i,1.0f);
		}
	}
	
	private HashTable timesVector (HashTable x){ // returns (Ax)
		// returns the HashTable y, with y(i)= <A(i,:),x>
		HashTable y = new HashTable(sizeHash);
		for (int i = 0; i<numRows; i++){
			if (rows[i]!=null){
				y.put(i,rows[i].dot(x));
			}
		}
		return y;
	}
	
	public void plus(SparseMatrix B) { // sets our matrix to (A+B)
		// elementwise addition of SparseMatrix's A and B
		for (int i=0; i<numRows; i++){
			if (B.rows[i]!=null){
				if (rows[i] == null){
					rows[i] = new HashTable(sizeHash);
				}
				rows[i].plus(B.rows[i]);
			}
		}
	}
	public void plus(int i, int j, float val){ // adds val to position (i,j)
		if (rows[i]==null){
			rows[i] = new HashTable(sizeHash);
		}
		rows[i].plus(j,val);
	}
	
	public SparseMatrix Transpose(){ // returns the transpose of A, (At)
		// returns the SparseMatrix with newMat(i,j) = A(j,i) for all i,j
		SparseMatrix newMat = new SparseMatrix(numRows);
		for (int i=0; i<numRows;i++){
			if (rows[i]!=null){
				transferNodesAndPr temp = rows[i].outNodesAndPr();
				for (int j=0 ; j<temp.keys.length; j++){
					newMat.put(temp.keys[j],i,temp.values[j]);
				}
			}
		}
		return newMat;
	}
	
	public SparseMatrix multiply(SparseMatrix B){ // returns (AB)
		// returns the sparseMatrix with each row equal 
		SparseMatrix Bt = B.Transpose();
		SparseMatrix newMat = new SparseMatrix(numRows);
		for (int i = 0; i<numRows; i++){
			if (Bt.rows[i]!=null){
				newMat.rows[i]=Bt.timesVector(rows[i]);
			}
		}
		return newMat;
	}
	
	private SparseMatrix multiply (float constant) { // returns (cA), and sets this to (cA)
		// **Both returns and sets this SparseMatrix**
		// Elementwise product by a constant
		for (int i = 0; i<numRows; i++){
			if (rows[i]!=null){
				rows[i].multiply(constant);
			}
		}
		return this;
	}
	
	public SparseMatrix multiplyTranspose(SparseMatrix B) { // returns (AXt)
		SparseMatrix newMat = new SparseMatrix(numRows);
		for (int i = 0; i<numRows; i++){
			newMat.rows[i] = timesVector(B.rows[i]);
		}
		return newMat;
	}
	
	public SparseMatrix AAt() { // returns (A At)
		// returns the matrix product of A and transpose(A)
		SparseMatrix newMat = multiplyTranspose(this);
		return newMat;
	}
	
	public SparseMatrix AtA() {// returns (At A)
		// returns the matrix product of tranpose(A) and A
		SparseMatrix newMat = this.Transpose().multiply(this);
		return newMat;
	}
	
	public SparseMatrix square(){ // returns (AA)
		// returns the matrix product of A and A
		SparseMatrix newMat = multiply(this);
		return newMat;
	}
	
	public SparseMatrix power(int k){ // returns (A^k)
		// returns the matrix power of A to the power k
		SparseMatrix newMat = new SparseMatrix(numRows);
		newMat.wipeIdentity();
		
		SparseMatrix nthPower = new SparseMatrix(numRows);
		nthPower.plus(this);
		
		while (k>0) {
			if ((k)%2==1){
				newMat = newMat.multiply(nthPower);
			}
			k = k/2;
			nthPower = nthPower.square();
		}
		return newMat;
	}
	
	public boolean hasDiag(){ // returns the boolean of (A has a non-zero diagonal element)
		for (int i=0; i<numRows; i++){
			if (get(i,i)!=0){
				return true;
			}
		}
		return false;
	}
	
	public int shortestLoop() { // returns the number of nodes of the shortest loop in A
		// assumes there exists no nodes w/o any out-edges
		
		// I am not completely sure if this is supposed to be the number of edges, or the number of nodes...
		
		// could be more efficient by either:
		// 1) checking existence of long term probabilities in submatrix
		//    while finding another method to check for cyclical nodes
		// 2) converting matrix to "dumb" matrix which only remembers if a element
		//    is zero or non-zero.
		// 3) itterating across all nodes, and doing breadth first search until that node is reached again.
		SparseMatrix newMat = new SparseMatrix(numRows);
		newMat.wipeIdentity();
		for (int i=1; i<=numRows+1; i++) {
			newMat = newMat.multiply(this);
			if (newMat.hasDiag()){
				return i;
			}
			++i;
		}
		return(-1); // if there are no loops
	}
	
	public float WalkPr(int[] W){
		float p = 1.0f;
		HashTable hashW = new HashTable(W.length/2+1);
		
		int prevValue = -1;
		for (int o : W){
			hashW.plus(o,1.0f);			
			// if any of these edges have 0 probability, return 0.0f
			if (prevValue>=0){
				if (get(prevValue,o) == 0.0f) {
					return 0.0f;
				}
			}
			prevValue = o;
		}
		
		for (HashEntry Q : hashW.table){
			HashEntry E = Q;
			while (E!=null){
				float alpha = computeAlpha(hashW, E.key);
				p*=alpha;
				E=E.next;
			}
		}
		return p;
	}
	
	private float computeAlpha (HashTable hashW, int v){
		if (rows[v]==null){
				rows[v] = new HashTable(sizeHash);
			}
		int OutWv = rows[v].numEntries; // pointer to next out-node in W
		int[] Ov = new int[OutWv];      // array of the out-nodes of v
		int Outv = 0;                   // pointer to next out-node not in W
		--OutWv;
		for (int p : rows[v].outNodes()) {
			if (hashW.get(p) == 0.0f) {
				Ov[Outv++]=p;
			}
			else{
				Ov[OutWv--]=p;
			}
		}
		
		float[][] r = new float[Outv+1][]; // (r) from paper
		r[0] = new float[1];
		r[0][0] = 1.0f;
		for (int i=1; i<=Outv; i++){
			r[i] = new float[i+1];
			r[i][0] = r[i-1][0]*(1-get(v,Ov[i-1]));
			r[i][i] = r[i-1][i-1]*get(v,Ov[i-1]);
			for (int j=1; j<i; j++){
				r[i][j] = r[i-1][j-1]*get(v,Ov[i-1])+r[i-1][j]*(1-get(v,Ov[i-1]));
			}
		}
		
		float alpha = 0.0f;		
		// compute alpha based on line (10) of WalkPr
		int p = (int)hashW.get(v);
		OutWv = rows[v].numEntries - Outv;  // re-use the reference to represent the number of outNodes in W
		for (int x=0; x<=Outv; x++){
			alpha += r[Outv][x]*power(inv(x+OutWv), p);
		}
		for (int i = Outv; i<Ov.length; i++){
			alpha*=get(v,Ov[i]);
		}
		return alpha;
	}
	
	public transferData TransPr(int K){
		// Disagreement 1: I think they forgot to account
		// for the possibility that a node loops back to itself
		// However, it may be by an assumption of the problem
		
		// I also neglected to sort the walks by the first and
		// last nodes. It seemed like a useless step.
		
		WalkNode[] WalkArray = new WalkNode[K]; // Array of walk values
		SparseMatrix[] pUkV = new SparseMatrix[K]; // k'th transition probability
		
		// compute W1
		pUkV[0] = new SparseMatrix(numRows);
		for (int i = 0; i<numRows; i++){
			for (int j=0; j<numRows; j++){
				int[] toPlaceW1 = new int[2];
				toPlaceW1[0] = i;
				toPlaceW1[1] = j;
				float Pr = WalkPr(toPlaceW1);
				if (Pr!=0.0f){
					pUkV[0].put(i,j,Pr);
					if (WalkArray[0] == null){
						WalkArray[0] = new WalkNode(toPlaceW1,Pr,1.0f);
					}
					else{
						WalkArray[0]=WalkArray[0].putPrev(toPlaceW1,Pr,1.0f);
					}
				}
			}
		}
		int L = shortestLoop();
		
		// compute W2 through WK
		for (int k=1; k<=K-1; k++){
			pUkV[k] = new SparseMatrix(numRows);
			WalkNode W = WalkArray[k-1];
			while (W!=null){
				int v = W.Walk[k];
				for (int w : rows[v].outNodes()){
					int[] toPlace = new int[k+2];
					HashTable walkHash = new HashTable(k/2+2);
					float Pr;
					float alpha;
					
					for (int i=0; i<=k; i++){
						toPlace[i] = W.Walk[i];
						walkHash.plus(W.Walk[i],1.0f);
					}
					toPlace[k+1] = w;
					
					// compute alpha and Pr
					if (k<L){
						Pr = W.walkPr*pUkV[0].get(v,w);
						alpha = 1.0f;
					}
					else{
						alpha = computeAlpha(walkHash,w);
						Pr = W.walkPr*alpha/W.alpha;
					}
					
					//write to file
					if (WalkArray[k]==null){
						WalkArray[k] = new WalkNode(toPlace,Pr,alpha);
					}
					else{
						WalkArray[k] = WalkArray[k].putPrev(toPlace,Pr,alpha);
					}
					pUkV[k].plus(W.Walk[0],w,Pr);
				}
				W=W.next;
			}
		}
		transferData Output = new transferData(WalkArray, pUkV);
		return Output;
	}
	
	public SparseMatrix SimRankRegular(int n,float c){
		// c : delay factor 0< c <1
		// n : number of steps
		// NOTE: I switched from AtA to AAt because
		// this is assumed to already be inverted
		
		// Disagreement 1: this transition probabilities are independent.
		// If one walk follows an edge, the same edge is instantiated in
		// the other walk.
		
		// Disagreement 2: the equation used does not work for s(u,u)
		SparseMatrix[] kProbs  = TransPr(n).kProbs;
		SparseMatrix Sn = new SparseMatrix(numRows);
		Sn.wipeIdentity();
		
		float constant=1.0f-c;
		
		for (int k=1; k<n; k++) {
			constant *= c;
			Sn.plus(kProbs[k-1].AAt().multiply(constant)); //For testing, currently AAt, should be AtA
			Sn.print(5);
			System.out.format("%d",k);
		}
		
		constant*=c/(1.0f-c);
		Sn.plus(kProbs[n-1].AAt().multiply(constant));//For testing, currently AAt, should be AtA
		return Sn;		
		
	}
	
	public float Sampling(int u, int v, int n, int N, float c){
		// c : delay factor 0< c <1
		// u : starting node 1
		// v : starting node 2
		// n : number of steps in a walk (n+1 nodes)
		// N : number of walks
		
		// I think they made some mistakes in their sampling method
		// if they are trying to get a truly random sample of the
		// random walks on the possible worlds, then they shouldnt
		// make two assumptions, one of which I will make since the
		// method breaks down if you don't assume it.
		
		// Assumption 1: When going through a random walk, you will
		// always have an out edge. Whenever they sample the possible
		// outedges, they consider the probability that there are no
		// no out edges, nor edges pointing back to itself to be 0.
		// However, I will assume this as well, since the method breaks
		// down without this assumption.
		
		// Assumption 2: When going through an node that they have
		// already visited, they don't take into consideration there
		// may be 2 or more out edges from this node. I won't make this
		// assumption. This will change the behavior of an edge leaving
		// an already visited node.
		
		// Disagreement 1: when approximating m^(k)(u,v) (eq. 13) they sum from
		// i=1 to N then they sum from j=1 to n. I think it should be
		// from j=0 to k. Otherwise you would get the same value for all
		// m^(k).
		
		// Disagreement 2: the definition for m^(k) (eq. 13) isn't consistant
		// with one of (m^(0),m^(n)). Since there are n+1 nodes in our n
		// length walk. Currently doing sum from m(0) to m(n), where
		// m(i) uses the first i+1 nodes.
		
		// Disagreement 3: the choice of a node based off of weighted choice is
		// not exactly correct. The use of an approximation makes sense
		// here.
		
		// Disagreement 4: the two random walks should end once they've met.
		// Under the s(u,v) = c/(|I(u)|+|I(v)|) * sum of (s(I_u,I_v))
		//         & s(u,u) = 1 	definition of simrank
		
		// edit: each random walk is paired with another 
		
				
		Random r = new Random();
		int[][] Uwalks = new int[N][];
		int[][] Vwalks = new int[N][];
		
		for (int i=0; i<N; i++){
			Uwalks[i] = sampleRandomWalk(u,n,r);
			Vwalks[i] = sampleRandomWalk(v,n,r);
		}
		float[] m = new float[n+1];
		for (int k=0; k<=n; k++){
			for (int i=0; i<N; i++){
				m[k] = m[k] + comp(Uwalks[i][k],Vwalks[i][k]);//L0DistanceToK(Uwalks[i],Vwalks[i],k);
			}
		}
		float constant=(1.0f-c)/N;
		float sn = m[0]*constant;
				
		for (int k=2; k<n; k++) {
			constant *= c;
			sn = sn + m[k-1]*constant;
		}
		constant = constant*c/(1.0f-c);
		sn = sn+m[n]*constant;	
		return sn;
	}
	public float Sampling2(int u, int v, int n, int N, float c){
		// c : delay factor 0< c <1
		// u : starting node 1
		// v : starting node 2
		// n : number of steps in a walk (n+1 nodes)
		// N : number of walks
		
		// Corrected a large error in Sampling. Still remain a few errors.
	
		Random r = new Random();
		int[] walkLog = new int[n+1];
		
		for (int i=0; i<N; i++){
			int result = sample2RandomWalks(u,v,n,r);
			if (result==1){
				walkLog[result] = walkLog[result]+1;
			}
		}
		float[] m = new float[n+1];
		for (int k=0; k<=n; k++){
			m[k] = walkLog[k]*1.0f;//L0DistanceToK(Uwalks[i],Vwalks[i],k);
		}
		float constant=(1.0f-c)/N;
		float sn = m[0]*constant;
				
		for (int k=2; k<n; k++) {
			constant = constant*c;
			sn = sn + m[k-1]*constant;
		}
		constant = constant*c/(1.0f-c);
		sn = sn+m[n]*constant;	
		return sn;
	}
	
	private  int[] sampleRandomWalk (int InitialPosition, int lenWalk, Random r) {
	
		int[] W = new int[lenWalk+1];
		SparseMatrix InstantiatedEdges = new SparseMatrix(numRows);

		W[0]=InitialPosition;
		
		for (int i=0; i<lenWalk; i++) {
			if (InstantiatedEdges.rows[W[i]]==null) {
				transferNodesAndPr TPN = rows[W[i]].outNodesAndPr();
				int[] OutNodes = TPN.keys;
				float[] OutVals = TPN.values;
				
				W[i+1] = OutNodes[pickRandom(OutVals,r)]; // weighted choice is inaccurate
														  // possible to find O(n^2) algorithm to replace
				InstantiatedEdges.put(W[i],W[i+1],1.0f);
			}
			else {
				transferNodesAndPr TPN = rows[W[i]].outNodesAndPr();
				int[] OutNodes = TPN.keys;
				float[] OutVals = TPN.values;
				float[] newOutVals = new float[OutNodes.length];
				for (int j=0; j<OutNodes.length; j++){ // this is where I modified their method
					// make already instantiated edges 1.0 probability
					newOutVals[j]=Math.max(OutVals[j],InstantiatedEdges.get(W[i],OutNodes[j]));
				}
				W[i+1] = OutNodes[pickRandom(newOutVals,r)];
				InstantiatedEdges.put(W[i],W[i+1],1.0f);
			}
		}

		return W;
	}
	private int sample2RandomWalks (int InitPos1, int InitPos2, int lenWalk, Random r) {

		int[] pos = {InitPos1, InitPos2};
		SparseMatrix InstantiatedEdges = new SparseMatrix(numRows);
		for (int i=0; i<lenWalk; i++) {
			if (pos[1]==pos[0]){
				return i;
			}
			for (int t=0; t<2; t++){
				if (InstantiatedEdges.rows[pos[t]]==null) {
					transferNodesAndPr TPN = rows[pos[t]].outNodesAndPr();
					int[] OutNodes = TPN.keys;
					float[] OutVals = TPN.values;
					int newpos = OutNodes[pickRandom(OutVals,r)]; // 
					InstantiatedEdges.put(pos[t],newpos,1.0f);
					pos[t] = newpos;
				}
				else {
					transferNodesAndPr TPN = rows[pos[t]].outNodesAndPr();
					int[] OutNodes = TPN.keys;
					float[] OutVals = TPN.values;
					float[] newOutVals = new float[OutNodes.length];
					for (int j=0; j<OutNodes.length; j++){ // this is where I modified their method
						// make already instantiated edges 1.0 probability. This isn't a perfect fix,
						// since posterior probability doesn't work out like that.
						newOutVals[j]=Math.max(OutVals[j],InstantiatedEdges.get(pos[t],OutNodes[j]));
					}
					int newpos = OutNodes[pickRandom(newOutVals,r)];
					InstantiatedEdges.put(pos[t],newpos,1.0f);
					pos[t] = newpos;
				}
			}
		}
		if (pos[1]==pos[0]){
			return lenWalk;
		}
		else{
			return -1;
		}
	}
	
	
	public float Mixed(int u, int v, int n, int N, float c, int L) {
		// c : delay factor 0< c <1
		// u : starting node 1
		// v : starting node 2
		// n : number of steps in a walk (n+1 nodes)
		// N : number of walks
		// L : length of walks where the sampling  method takes over
		
		Random r = new Random();
		int[][] Uwalks = new int[N][];
		int[][] Vwalks = new int[N][];
		
		for (int i=0; i<N; i++){
			Uwalks[i] = sampleRandomWalk(u,n,r);
			Vwalks[i] = sampleRandomWalk(v,n,r);
		}
		float[] m = new float[n-L];
		for (int k=L+1; k<=n; k++){
			for (int i=0; i<N; i++){
				m[k-L-1] = m[k-L-1] + comp(Uwalks[i][k],Vwalks[i][k]);//L0DistanceToK(Uwalks[i],Vwalks[i],k);
			}
		}
		
		float sn = SimRankRegular(L,c).get(u,v);
		
		float constant=(1.0f-c)*power(c,L+1)/N;
				
		for (int k=0; k<n-L-1; k++) {
			constant *= c;
			sn = sn + m[k]*constant;
		}
		
		
		constant = constant*c/(1.0f-c);
		sn = sn+m[n-1-L]*constant;		
		return sn;
	}
	
	public void print(int floatingPoint){
		// only built to handle graphs up to size 100
		System.out.print("\t SparseMatrix \n");
		
		char[] header = new char[numRows*floatingPoint+3*numRows+4];
		header[0] = ' '; header[1] = ' '; header[2] = ' '; header[4] = ' ';
		for (int i = 4; i<numRows*floatingPoint+3*numRows+4; i++){
			String str = String.valueOf((i-4)/(floatingPoint+3));
			if ((i-4)%(floatingPoint+3)==0){
				for (int j = 0; j<str.length(); j++){
					i=i+j;
					header[i] = str.charAt(j);
				}
			}
			else{
				header[i] = ' ';
			}
		}
		header[numRows*floatingPoint+3*numRows+3] = '\n';
		System.out.print(header);
		
		for (int i=0; i<numRows; i++){
			char[] line = new char[numRows*floatingPoint+3*numRows+4];
			String strOfI = String.valueOf(i);
			int next = 0;
			for (int j=0; j<strOfI.length(); j++){
				line[next] = strOfI.charAt(j);
				++next;
			}
			while (next<4){
				line[next] = ' ';
				next++;
			}
			for (int j=0; j<numRows; j++){
				float val = get(i,j);
				String str = String.valueOf(val);
				int count = 0;
				for (int k = 0; k < str.length(); k++){
					if (count<floatingPoint+2 && val!=0.0f){
						char o = str.charAt(count);
						line[next] = o;
						++count;
						++next;
					}
				}
				while (count<floatingPoint+2){
					if (val!=0.0f){
						line[next] = '0';
					}
					else{
						line[next] = ' ';
					}
					++count;
					++next;
				}
				line[next] = '|';
				++next;
			}
			line[next-1]='\n';
			String outLine = new String(line);
			System.out.print(outLine);
		}
	}
	
	

	
   ///////////////////Static methods and SubClasses////////////////////	
	private static float L0DistanceToK(int[] A, int[] B,int k){
		float sum = 0;
		for (int i = 0; i<=k; i++){
			if (A[i]==B[i]){
				sum = sum +1.0f;
			}
		}
		return sum;
	}
	
	private static float comp(int A, int B){
		if (A==B){
			return 1.0f;
		}
		else{
			return 0.0f;
		}
	}
	
	
	private static int pickRandom(float[] floats, Random r){
		float sum = 0.0f;
		for (float o: floats){
			sum+=o;
		}

		float randVal = sum*r.nextFloat();
		for (int i=0;i<floats.length;i++){
			randVal -= floats[i];
			if (randVal<0.0f) {
				return i;
			}

		}
		return floats.length-1;
	}
	
	private static float inv(int x){
		if (x==0) {
			return 1.0f;
		}
		else{
			return 1.0f/x;
		}
	}
	
	public static int log10 (int x){
		int count = 0;
		while (x/10>0){
			++count;
			x/=10;
		}
		return count;
	}
	
	private static float power(float x, int k) {
		float output = 1.0f;
		
		while (k>0) {
			if ((k-1)%2==0){
				output = output*x;
				--k;
			}
			k/=2;
			x = x*x;
		}
		return output;
	}

	
	
		
	private class Node{
		Node next;
		int key;
		
		public Node(int k){
			key = k;
			next = null;
		}
		
		public void put(int k){
			if (next == null){
				next = new Node(k);
			}
			else{
				next.put(k);
			}
		}
	}
}


class WalkNode{
	public WalkNode next;
	public int[] Walk;
	public float alpha;
	public float walkPr;
	
	public WalkNode(int[] W,float p, float a){
		Walk = W;
		alpha = a;
		walkPr = p;
		next = null;
	}
	public WalkNode(){
		Walk = null;
		alpha = 1.0f;
		walkPr = 1.0f;
		next = null;
	}
	public void put(int[] W, float p, float a){
		if (next == null){
			next = new WalkNode(W,p,a);
		}
		else{
			next.put(W,p,a);
		}
	}
	public WalkNode putPrev(int[] W,float p, float a){
		WalkNode prev = new WalkNode(W,p,a);
		prev.next = this;
		return prev;
	}
}

class transferData{
	public WalkNode[] Walks;
	public SparseMatrix[] kProbs;
	
	public transferData(WalkNode[] Ws, SparseMatrix[] kPs){
		Walks = Ws;
		kProbs = kPs;
	}
	
	public void print(int logVal){
		System.out.print("\t Walks and Probability\n");
		int nextChar;
		int count;
		for (WalkNode W: Walks){
			WalkNode E = W;
			while (E!=null){
				char[] line = new char[E.Walk.length*(logVal+2)+2];
				nextChar = 0;
				for (int o : E.Walk){
					line[nextChar++] = ' ';
					String str = String.valueOf(o);
					for (int j=0; j<str.length(); j++){
						line[nextChar++]=str.charAt(j);
					}
					while(nextChar%(logVal+2)!=0){
						line[nextChar++] = ' ';
					}
				}
				line[nextChar++] = ' ';
				line[nextChar++] = ' ';
				System.out.print(line);
				System.out.println(E.walkPr);
				E = E.next;
			}
		}
	}
}
