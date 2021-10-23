from utils import *


def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts))-1
    return tf.one_hot(X, len(tokenizer.word_index))


def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def complete_text(text, n_chars=100, temperature=1):
    print()
    print(text, end="")
    for _ in range(n_chars):
        next_ch = next_char(text, temperature)
        print(next_ch, end="")
        text += next_ch
    print()
    return text


filename = input('Enter file name on which model was trained : ')
with open(f"datasets\\{filename}.txt") as file:
    text = file.read()
print("Total character count in data trained on : ", len(text))
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
base_model = find_base(filename)
load_filepath = loading_path(filename, base_model)
print()

model = keras.models.load_model(load_filepath, compile=False)
print('Do you want to view the model summary?')
print('1 - Yes')
print('2 - No')
condition = int(input('Enter your choice : '))
if condition == 1:
    model.summary()

print()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
input_text = input('Enter some intial text as context : ')
n_chars = int(input('Enter number of characters to generate : '))
text = complete_text(input_text, n_chars)
print()
print("If the text is satisfactory consider saving it in results")
print('1 - save')
print('0 - exit')

if int(input('Enter : ')) == 1:
    filename = input('Enter filename : ')
    with open(f"results\\{filename}.txt", "w") as file:
        file.write(f"Context : {input_text}\n")
        file.write(f"Characters generated : {n_chars}\n")
        file.write("\n")
        file.write("Generated text : ")
        file.write("\n\n")
        file.write(text)
