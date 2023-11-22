# Nepali RoBERTa Model

This repository contains code to train a Byte-Level Byte Pair Encoding (BPE) tokenizer for Nepali text using the Hugging Face `tokenizers` library. Additionally, it provides code to train a RoBERTa model for masked language modeling tasks using the tokenizer.

#### Requirements
- Python 3.x
- `transformers` library
- `tokenizers` library
- `torch` library
- `datasets` library

#### Usage

1. **Clone the Repository**
```bash
git clone https://github.com/bkhanal-11/nepali-roberta
cd nepali-roberta
```
2. **Download Dataset** 

The dataset is present in the google drive in this [link](https://drive.google.com/drive/folders/1evlBhsOKxDGEeD3AxDoKUUCfxSh6V9rc?usp=sharing). The original dataset has been chunked down for easier storage.

3. Install Dependencies

```bash
pip3 install -r requirements.txt
```

4. Training the Tokenizer

Move the dataset in the nepali-text directory. Modify the parameters in train_tokenizer.py as needed.
Run the training script:

```bash
python3 train_tokenizer.py
```

5. Training the RoBERTa Model

Use the trained tokenizer to tokenize the Nepali text data. Modify parameters in train_roberta.py if required. Run the RoBERTa training script:
```bash
python train_roberta.py
```

6. Using the Trained Model

After training, utilize the trained RoBERTa model for tasks like masked language modeling, text generation, etc.

A sample example from the tokenizer is given below:

```python
sentence2 = "सोमबार १११औँ अन्तर्राष्ट्रिय श्रमिक महिला दिवसको सन्दर्भमा अनेरास्ववियूले आयोजना गरेको टेम्पो चालक महिला सम्मान कार्यक्रमलाई सम्बोधन गर्दै भुसालले ५० प्रतिशत भन्दा बढी संख्यामा रहेका महिलाहरुले सबै क्षेत्रमा ५० प्रतिशतभन्दा बढी अधिकार प्राप्तिको निम्ति"
encoded_input = tokenizer.encode(sentence2)
tokenizer.decode(encoded_input.ids)
```

Output:
```bash
'सोमबार १११औँ अन्तर्राष्ट्रिय श्रमिक महिला दिवसको सन्दर्भमा अनेरास्ववियूले आयोजना गरेको टेम्पो चालक महिला सम्मान कार्यक्रमलाई सम्बोधन गर्दै भुसालले ५० प्रतिशत भन्दा बढी संख्यामा रहेका महिलाहरुले सबै क्षेत्रमा ५० प्रतिशतभन्दा बढी अधिकार प्राप्तिको निम्ति'
```