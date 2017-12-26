

class HashTable{
	private final int tableSize;
	public int numEntries;
	public HashEntry[] table;
	
	public HashTable(int s){
		numEntries = 0;
		tableSize = s;
		table = new HashEntry[tableSize];
	}
	
	public void clear(){
		for (int i=0; i<tableSize; i++){
			table[i]=null;
		}
		numEntries = 0;
	}
	
	public float get(int key){
		int _key_ = hash(key) % tableSize;
		HashEntry E = table[_key_];		
		
		while (E!=null){
			if (E.key == key){
				return E.value;
			}
			E=E.next;
		}
		return 0.0f;
	}
	
	public void put(int key, float value){
		int _key_ = hash(key)% tableSize;
		HashEntry E = table[_key_];
		if (value==0.0f){
			remove(key);
		}
		else if (E==null){
			table[_key_] = new HashEntry(key,value);
			++numEntries;
		}
		else if (E.put(key, value)){
			++numEntries;
		}
	}
	
	public void remove(int key){
		int _key_ = hash(key)%tableSize;
		HashEntry E = table[_key_];
		if (E!=null){
			if (E.key == key){
				table[_key_] = E.next;
				--numEntries;
			}
			else{
				HashEntry nextE = E.next;
				while (nextE!=null){
					if (nextE.key == key){
						E.next = nextE.next;
						--numEntries;
						nextE = null;
					}
					else{
						E = nextE;
						nextE = E.next;
					}
				}
			}
		}
	}
	
	public void plus(int key, float value){
		int _key_ = hash(key)% tableSize;
		if (value == 0.0f){
			;
		}
		HashEntry E = table[_key_];
		if (E==null){
			table[_key_] = new HashEntry(key,value);
			++numEntries;
		}
		else if (E.plus(key,value)){
			++numEntries;
		}
	}
	
	public void plus(HashTable X) {
		for (HashEntry o : X.table) {
			HashEntry E2 = o;
			while (E2!=null) {
				int key2 = E2.key;
				int _key_ = hash(key2)%tableSize;
				HashEntry E = table[_key_];
				if (E==null) {
					table[_key_] = new HashEntry(key2,E2.value);
					++numEntries;
				}
				else if (E.plus(key2,E2.value)){
					++numEntries;
				}
				E2 = E2.next;
			}
		}
	}
	
	public int[] outNodes(){
		int[] outKeys = new int[numEntries];
		int i = 0;
		for (HashEntry o : table) {
			HashEntry E = o;
			while (E!=null) {
				outKeys[i++]=E.key;
				E = E.next;
			}
		}
		return outKeys;
	}
	
	public transferNodesAndPr outNodesAndPr(){
		int[] outKeys = new int[numEntries];
		float[] outVals = new float[numEntries];
		int i = 0;
		for (HashEntry o : table) {
			HashEntry E = o;
			while (E!=null) {
				outVals[i] = E.value;
				outKeys[i++] = E.key;
				E = E.next;
			}
		}
		transferNodesAndPr output = new transferNodesAndPr(outKeys,outVals);
		return output;
	}

	public void multiply(float c) {
		for (HashEntry o : table){
			HashEntry E = o;
			while (E!=null) {
				E.multiply(c);
				E = E.next;
			}
		}
	}
	
	public float dot(HashTable X) {
		if (X==null){
			return 0.0f;
		}
		if  (X.tableSize!=tableSize){
			System.out.println("ERROR: Dimensions disagree for dot product");
			return 0.0f;
		}
		float sum = 0.0f;
		
		for (int i=0 ; i<tableSize; i++) {
			HashEntry E1 = table[i];
			HashEntry E2 = X.table[i];
			
			if (E1!=null && E2!=null){
				while (E1!=null){
					while (E2!=null){
						if (E1.key == E2.key) {
							sum+=E1.value*E2.value;
						}
						E2=E2.next;
					}
					E1=E1.next;
				}
			}
		}
		return sum;
	}
		
	private static int hash(int key){
		return key; //apply some function if problems arise
	}

}

class transferNodesAndPr {
	public int[] keys;
	public float[] values;
	
	transferNodesAndPr(int[] k, float[] v) {
		keys = k;
		values = v;
	}
}

class HashEntry {
	public int key;
	public float value;
	public HashEntry next;
	
	public HashEntry(int k, float val) {
		key = k;
		value = val;
		next = null;
	}
	
	public HashEntry(int k){
		key = k;
		value = 1.0f;
		next = null;
	}
	
	// I vaugly remember that recursion like this is relatively
	// slow so if this would be a problem, I can revert it back
	// to the old version
	public boolean put(int k, float c){
		if (k==key){
			value = c;
			return false;
		}
		else if (next==null){
			next = new HashEntry(k,c);
			return true;
		}
		else{
			return next.put(k,c);
		}
	}
	
	public boolean plus(int k, float c){
		if (k==key){
			value=value+c;
			return false;
		}
		else if (next==null){
			next = new HashEntry(k,c);
			return true;
		}
		else{
			return next.plus(k,c);
		}
	}
	
	public void multiply(float c){
		value *=c;
	}
	
}



