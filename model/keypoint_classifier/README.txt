1. keypoint.csv (The Notebook) - 
	Kya hai?: Ye ek raw data file hai.
	Kaam: Jab aapne logging_csv wala function chala kar 'k' dabaya tha, to sara data isi file mein gaya tha.
	Structure: Isme har line aisi dikhti hai: ID, x1, y1, x2, y2....
		Yani, "Gesture ID 0 (Open Palm) ke liye 21 points ki location ye hai."
	Asli Matlab: Ye wo "Exercise Book" hai jisme aapne AI ko sikhane ke liye bohot saare examples likhe hain.
	
	

2. keypoint_classifier_label.csv (The Dictionary)
	Kya hai?: Ye ek simple text file hai.
	Kaam: Isme sirf gestures ke naam likhe hote hain (jaise: Open, Close, Pointer).
	Kyun?: AI model sirf numbers (0, 1, 2) samajhta hai. Ye file un numbers ko Names mein badalti hai taaki aapko screen par "Open" likha dikhe, na ki sirf "0".
	
	

3. keypoint_classifier.tflite (The Trained Brain)
	Kya hai?: Ye asli AI Model hai (TensorFlow Lite format mein).
	Kaam: Ye wahi file hai jo input (21 points) leti hai aur seconds ke hazarve hisse mein batati hai ki ye kaunsa gesture hai.
	Kyun?: Ye .tflite format mein hai kyunki ye bahut fast aur halki (lightweight) hoti hai, jo laptop ya mobile par live video ke saath bina ruke (lag-free) chal sakti hai.
	

4. keypoint_classifier.py (The Interpreter)
	Kya hai?: Ye ek Python script hai jo .tflite model ko chalaane ka kaam karti hai.
	Kaam: Ye .tflite file ko load karti hai.
		Hath ke points ko model ke hisab se "Input" deti hai.
		Model se "Prediction" leti hai aur use wapas main.py ko bhejti hai.
	Asli Matlab: Ye AI model aur aapke code ke beech ka Manager hai.
	

5. keypoint_classification.ipynb (The Teacher)
	Kya hai?: Ye ek Jupyter Notebook file hai.
	Kaam: Is file ka use karke model ko Train kiya gaya hai.
		Ye keypoint.csv se data padhti hai.
		Use "Neural Network" sikhata hai.
		Aur aakhir mein keypoint_classifier.tflite file ko generate (paida) karti hai.
	Note: Jab aapka program chal raha hota hai, is file ki zarurat nahi hoti. Ye sirf tab kaam aati hai jab aap koi Naya Gesture sikhana chahte hain.

	
