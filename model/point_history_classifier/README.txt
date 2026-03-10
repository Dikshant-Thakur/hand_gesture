1. point_history.csv (The Path Recorder)
	Kya hai?: Ye un 16 pichle points ka data hai jo aapki index finger ki tip ne chode hain.
	Kaam: Jab aap movement wala data record karte hain, toh ye (x, y) coordinates ki ek lambi ladi (sequence) save karta hai.
	Farq: keypoint.csv mein ek hi frame ke 21 joints the. Isme ek hi point (finger tip) ke 16 frames ka itihaas (history) hota hai.
	
2. point_history_classifier_label.csv (Movement Names)
	Kya hai: Labels ki file.
	Contents: Isme "Static", "Clockwise", "Counter-Clockwise", ya "Stop" jaise naam hote hain.
	Kaam: AI jab result mein 2 dega, toh ye file batayegi ki 2 ka matlab "Clockwise Circle" hai.
	
3. point_history_classifier.tflite (The Movement Brain)
	Kya hai: Ye ek alag AI model hai.
	Kaam: Ye "Pattern Recognition" karta hai. Iska dimaag is tarah se train kiya gaya hai ki agar points gol ghoom rahe hain, toh ye use "Circle" pehchan le.
	Specialty: Ye Time (waqt) ko samajhta hai. Isse pata hai ki point pehle kahan tha aur ab kahan hai.
	
4. point_history_classifier.py (The Coordinator)
	Kya hai: Wahi Python script jo humne pehle dekhi thi.
	Kaam: 
		1. Ye .tflite model ko load karti hai.
		2. point_history (16 points ki list) ko model ko deti hai.
		3. Model se prediction mangti hai.

5. point_history_classification.ipynb (The Training Lab)
	Kya hai: Jupyter Notebook.
	Kaam: Iska use karke point_history.csv ke data ko train kiya gaya hai. Isme aksar LSTM ya Simple RNN (Neural Networks jo sequence samajhte hain) ka use hota hai.
