% Creating a MATLAB GUI
fig = uifigure('Name', 'Cryptographic GUI App', 'Position', [100, 100, 400, 300]);

% Creating a menu bar
menuBar = uimenu(fig, 'Text', 'Weekly Assessments');

% Week 1 - Classic Cryptography
week1Menu = uimenu(menuBar, 'Text', 'Week 1 - Classic Cryptography');
uimenu(week1Menu, 'Text', 'Caesar Cipher', 'MenuSelectedFcn', @(src, event) caesarCipher());
uimenu(week1Menu, 'Text', 'cryptoanalysis', 'MenuSelectedFcn', @(src, event) cryptoanalysis());
uimenu(week1Menu, 'Text', 'analyzeLetterDistribution', 'MenuSelectedFcn', @(src, event) analyzeLetterDistribution());
uimenu(week1Menu, 'Text', 'vectorizationLetterDistribution', 'MenuSelectedFcn', @(src, event) vectorizationLetterDistribution());

% Week 2 - Random Number Generators
week2Menu = uimenu(menuBar, 'Text', 'Week 2 - Random Number Generators');
uimenu(week2Menu, 'Text', 'Values 4x32', 'MenuSelectedFcn', @(src, event) philoxGenerator());
uimenu(week2Menu, 'Text', 'Random Values', 'MenuSelectedFcn', @(src, event) simdMersenneTwister());

% Week 3 - Bitwise Manipulation
week3Menu = uimenu(menuBar, 'Text', 'Week 3 - Bitwise Manipulation');
uimenu(week3Menu, 'Text', 'XOR Encryption', 'MenuSelectedFcn', @(src, event) xorEncryption());

% Week 4 - Block Ciphers
week4Menu = uimenu(menuBar, 'Text', 'Week 4 - Block Ciphers');
uimenu(week4Menu, 'Text', 'Key Generation', 'MenuSelectedFcn', @(src, event) keyGeneration());
uimenu(week4Menu, 'Text', 'DES Algorithm', 'MenuSelectedFcn', @(src, event) desAlgorithm());

% Week 5 - Stream Ciphers
week5Menu = uimenu(menuBar, 'Text', 'Week 5 - Stream Ciphers');
uimenu(week5Menu, 'Text', 'RC4 Cipher', 'MenuSelectedFcn', @(src, event) rc4Cipher());
uimenu(week5Menu, 'Text', 'XOR stream cipher', 'MenuSelectedFcn', @(src, event) xorCipher());
uimenu(week5Menu, 'Text', 'Custom Stream Cipher', 'MenuSelectedFcn', @(src, event) streamCipher());

% Week 6 - Hashing and CRC
week6Menu = uimenu(menuBar, 'Text', 'Week 6 - Hashing and CRC');
uimenu(week6Menu, 'Text', 'Hashing Routine', 'MenuSelectedFcn', @(src, event) hashingRoutine());
uimenu(week6Menu, 'Text', 'CRC Routine', 'MenuSelectedFcn', @(src, event) crcRoutine());

% Week 7 - HMAC Hashing
week7Menu = uimenu(menuBar, 'Text', 'Week 7 - HMAC Hashing');
uimenu(week7Menu, 'Text', 'HMAC Hashing', 'MenuSelectedFcn', @(src, event) hmacHashing());

% Week 8 - Authenticated Encryption
week8Menu = uimenu(menuBar, 'Text', 'Week 8 - Authenticated Encryption');
uimenu(week8Menu, 'Text', 'GCM Block Cipher', 'MenuSelectedFcn', @(src, event) gcmBlockCipher());

% Week 9 - Algorithmic Complexity
week9Menu = uimenu(menuBar, 'Text', 'Week 9 - Algorithmic Complexity');
uimenu(week9Menu, 'Text', 'Growth Rates Visualization', 'MenuSelectedFcn', @(src, event) growthRatesVisualization());

% Week 10 - RSA Encryption
week10Menu = uimenu(menuBar, 'Text', 'Week 10 - RSA Encryption');
uimenu(week10Menu, 'Text', 'RSA Algorithm', 'MenuSelectedFcn', @(src, event) rsaAlgorithm());


% Define callback functions for each menu option
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function caesarCipher()
    % Implement Caesar Cipher functionality################################
    plain_text = 'My Name Is Mark';
method = 'caesar';
key = 3; 

encrypted_text = classic_cryptography(plain_text, method, key);
disp(['Encrypted text: ', encrypted_text]);

function encrypted_text = classic_cryptography(plain_text, method, key)
    % Converting the input text to uppercase
    plain_text = upper(plain_text);

    % Checking the chosen method
    switch method
        case 'caesar'
            encrypted_text = caesar_cipher(plain_text, key);
        case 'vigenere'
            encrypted_text = vigenere_cipher(plain_text, key);
        case 'one_time_pad'
            encrypted_text = one_time_pad(plain_text, key);
        otherwise
            error('Invalid method. Supported methods: caesar, vigenere, one_time_pad');
    end
end

% Caesar Cipher
function encrypted_text = caesar_cipher(plain_text, shift)
    encrypted_text = '';
    for i = 1:length(plain_text)
        if isletter(plain_text(i))
            encrypted_text = [encrypted_text, char(mod(plain_text(i) - 'A' + shift, 26) + 'A')];
        else
            encrypted_text = [encrypted_text, plain_text(i)];
        end
    end
end

% Vigenere Cipher
function encrypted_text = vigenere_cipher(plain_text, key)
    encrypted_text = '';
    for i = 1:length(plain_text)
        if isletter(plain_text(i))
            shift = mod(key(mod(i - 1, length(key)) + 1) - 'A', 26);
            encrypted_text = [encrypted_text, char(mod(plain_text(i) - 'A' + shift, 26) + 'A')];
        else
            encrypted_text = [encrypted_text, plain_text(i)];
        end
    end
end

% One-Time Pad
function encrypted_text = one_time_pad(plain_text, key)
    if length(key) < length(plain_text)
        error('Key length is not sufficient for the given plain text.');
    end

    encrypted_text = '';
    for i = 1:length(plain_text)
        if isletter(plain_text(i))
            encrypted_text = [encrypted_text, char(mod(plain_text(i) - 'A' + key(i) - 'A', 26) + 'A')];
        else
            encrypted_text = [encrypted_text, plain_text(i)];
        end
    end
end
end
function cryptoanalysis()
    % Implement cryptoanalysis functionality################################
    datasetPath = '/MATLAB Drive/english_monograms.csv';

% Loading the dataset
data = readtable(datasetPath, 'Delimiter', '\t', 'ReadVariableNames', false);

disp(data);

% combined column into separate columns
splitData = cellfun(@(x) strsplit(x, ','), data.Var1, 'UniformOutput', false);
splitData = vertcat(splitData{:});

% Extracting letters and frequencies
letters = splitData(:, 1);
frequencies = str2double(splitData(:, 2));

% bar graph
bar(letters, frequencies);
xlabel('Letters');
ylabel('Frequency (%)');
title('English Letter Distribution');
grid on;
end

function analyzeLetterDistribution()
    % Implement analyzeLetterDistribution functionality################################
    % Specifying the path to the dataset
    datasetPath = '/MATLAB Drive/english_monograms.csv';

    % Loading the dataset
    data = readtable(datasetPath, 'Delimiter', '\t', 'ReadVariableNames', false);

    splitData = cellfun(@(x) strsplit(x, ','), data.Var1, 'UniformOutput', false);
    splitData = vertcat(splitData{:});

    letters = splitData(:, 1);
    frequencies = str2double(splitData(:, 2));

    % Accepting user input
    inputText = upper(input('Enter an English language sentence or paragraph: ', 's'));

    % Counting occurrences of each letter in the input text
    letterCounts = zeros(size(letters));
    for i = 1:length(letters)
        letterCounts(i) = sum(inputText == letters{i});
    end

    % Normalizing the counts to percentages
    letterPercentages = (letterCounts / length(inputText)) * 100;

    % bar graph
    figure;
    bar(letters, letterPercentages);

    xlabel('Letters');
    ylabel('Frequency (%)');
    title('Letter Distribution in Input Text');
    grid on;
end

function vectorizationLetterDistribution()
    % Implement vectorizationLetterDistribution functionality################################
    % Specifying the path to the dataset
    datasetPath = '/MATLAB Drive/english_monograms.csv';

    % Loading the dataset
    data = readtable(datasetPath, 'Delimiter', '\t', 'ReadVariableNames', false);

    splitData = cellfun(@(x) strsplit(x, ','), data.Var1, 'UniformOutput', false);
    splitData = vertcat(splitData{:});

    letters = splitData(:, 1);
    frequencies = str2double(splitData(:, 2));

    inputText = upper(input('Enter an English language sentence or paragraph: ', 's'));

    letterCounts = zeros(size(letters));
    for i = 1:length(letters)
        letterCounts(i) = sum(strcmp(inputText, letters{i}));
    end

    letterPercentages = (letterCounts / length(inputText)) * 100;

    figure;
    bar(letters, letterPercentages);

    xlabel('Letters');
    ylabel('Frequency (%)');
    title('Letter Distribution in Input Text');
    grid on;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function philoxGenerator()
    % Implement Philox 4x32 generator functionality
    % random number generators
rng_default = RandStream('mt19937ar');
rng_dsfmt19937 = RandStream('dsfmt19937', 'Seed', 'shuffle');
rng_philox = RandStream('philox4x32_10', 'Seed', 'shuffle');

% One-Time Pad with a randomly chosen generator
rng_selected = randi([1, 3]); % Choose a random generator (1, 2, or 3)
rng_selected_stream = {rng_default, rng_dsfmt19937, rng_philox};
selected_rng = rng_selected_stream{rng_selected};

plain_text_example = 'HELLO';
encrypted_text_example = one_time_pad_char(double(plain_text_example), selected_rng);
disp(['Plain Text: ', plain_text_example]);
disp(['Encrypted Text: ', encrypted_text_example]);

% random numbers using different generators
random_values_default = rand(rng_default, 10000, 1);
random_values_dsfmt19937 = rand(rng_dsfmt19937, 10000, 1);
random_values_philox = rand(rng_philox, 10000, 1);

% output of different generators
figure;
subplot(3,1,1);
histogram(random_values_default, 'DisplayName', 'Mersenne Twister');
title('Mersenne Twister');

subplot(3,1,2);
histogram(random_values_dsfmt19937, 'DisplayName', 'SIMD-oriented Fast Mersenne Twister');
title('SIMD-oriented Fast Mersenne Twister');

subplot(3,1,3);
histogram(random_values_philox, 'DisplayName', 'Philox 4x32 with 10 rounds');
title('Philox 4x32 with 10 rounds');
xlabel('Random Values');

% One-Time Pad implementation using a randomly chosen generator
function encrypted_text = one_time_pad_char(plain_text, rng)
    plain_text = upper(plain_text);
    key = randi(rng, [1, 26], size(plain_text)) - 1; 
    encrypted_text = char(mod(plain_text - 'A' + key, 26) + 'A');
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % week 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xorEncryption()
    % Implement XOR Encryption functionality
    % XOR encryption with a key
plaintext = 'secrets';
key = 'EDR141'; 

% XOR encryption
encrypted_binary = xor_encrypt(plaintext, key);

% Encrypted binary with the same key again
decrypted_binary = xor_encrypt(encrypted_binary, key);

decrypted_str = char(bin2dec(reshape(char(decrypted_binary + '0'), 8, []).'));

% Decrypted string
disp('Decrypted String:');
disp(decrypted_str);

% Decrypted string matches the original plaintext
disp('Is Decryption Correct?');
disp(strcmp(plaintext, decrypted_str));

% XOR encryption at bitwise level
function encrypted_binary = xor_encrypt(plaintext, key)
    binary = reshape(dec2bin(plaintext, 8).' - '0', 1, []);

    key = repmat(key, 1, ceil(length(binary) / length(key)));

    % XOR operation
    encrypted_binary = xor(binary, key(1:length(binary)));

    disp('Encrypted Binary:');
    disp(encrypted_binary);

    encrypted_str = char(bin2dec(reshape(char(encrypted_binary + '0'), 8, []).'));

    disp('Encrypted String:');
    disp(encrypted_str);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function keyGeneration(~, ~)
    % Implement key Generation functionality################################
    % Prompt for input values
    n = input('Enter the number of keys (default is 1): ');
    m = input('Enter the number of initialization vectors (default is 1): ');

    % Validate and set default values
    if isempty(n) || ~isnumeric(n) || n <= 0
        n = 1;
    end

    if isempty(m) || ~isnumeric(m) || m <= 0
        m = 1;
    end

    % Call the modified keygen function
    keys = keygen(n, m);

    % Display the generated keys
    disp('Generated Keys:');
    disp(keys);
end

function keys = keygen(n, m)
    keys = cell(n, m);

    for i = 1:n
        for j = 1:m
            key = round(rand(8, 7));
            key(:, 8) = mod(sum(key, 2) + 1, 2);
            keys{i, j} = key;

            if n > 1
                keys{i, j + 1} = round(rand(8, 8)); % Initialization Vectors
            end
        end
    end
end

function desAlgorithm()
    % Implement DES Algorithm functionality################################
    % DES algoritm
plaintext = 'hellow DES';
shift = 3;
encrypted_text = caesar_cipher(plaintext, shift);

disp(['Plaintext: ', plaintext]);
disp(['Encrypted text: ', encrypted_text]);


function encrypted_text = caesar_cipher(plaintext, shift)
    % Convert the plaintext to uppercase
    plaintext = upper(plaintext);
    
    % Initialize the encrypted text
    encrypted_text = '';
    
    % Iterate through each character in the plaintext
    for i = 1:length(plaintext)
        % Check if the character is a letter
        if isletter(plaintext(i))
            % Shift the letter by the specified amount
            encrypted_char = char(mod(double(plaintext(i)) - 65 + shift, 26) + 65);
        else
            % If the character is not a letter, leave it unchanged
            encrypted_char = plaintext(i);
        end
        
        % Append the encrypted character to the result
        encrypted_text = [encrypted_text, encrypted_char];
    end
end
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rc4Cipher()
    % Implement RC4 Cipher functionality################################
    % RC4 cipher
plaintext = 'RC4 cipher implementation';
key = 'secretkey';

encrypted_text_rc4 = rc4_cipher(plaintext, key);

disp('Original Text:');
disp(plaintext);
disp('Encrypted Text (RC4):');
disp(encrypted_text_rc4);



function encrypted_text = rc4_cipher(plaintext, key)
    S = rc4_key_schedule(key);
    
    % Text to ASCII values
    text_ascii = double(plaintext);
    
    % Encryption using RC4
    encrypted_ascii = rc4_process(text_ascii, S);
    
    % Convertion back to characters
    encrypted_text = char(encrypted_ascii);
end

function S = rc4_key_schedule(key)
    key_length = length(key);
    
    % Initialization S array
    S = 0:255;
    
    % Key-scheduling algorithm
    j = 0;
    for i = 1:256
        j = mod(j + S(i) + key(mod(i - 1, key_length) + 1), 256) + 1;
        % Swap S(i) and S(j)
        temp = S(i);
        S(i) = S(j);
        S(j) = temp;
    end
end

function output = rc4_process(input, S)
    % Initialization
    i = 1;
    j = 0;
    
    % Pseudo-random generation algorithm
    output = input;
    for k = 1:length(input)
        i = mod(i + 1, 256);
        j = mod(j + S(i + 1), 256);
        
        % Swap S(i) and S(j)
        temp = S(i + 1);
        S(i + 1) = S(j + 1);
        S(j + 1) = temp;
        
        % Pseudo-random byte
        t = mod(S(i + 1) + S(j + 1), 256);
        output(k) = bitxor(input(k), S(t + 1));
    end
end
end

function xorCipher()
    % Implement XOR stream cipher functionality################################
    % XOR stream cipher
plaintext = 'XOR stream cipher';
key = 'secretkey';
encrypted_text = xor_stream_cipher(plaintext, key);
disp('Original Text:');
disp(plaintext);
disp('Encrypted Text:');
disp(encrypted_text);

function encrypted_text = xor_stream_cipher(plaintext, key)
    % Text and key to ASCII values
    text_ascii = double(plaintext);
    key_ascii = double(key);
    
    % Key is as long as the plaintext
    key_repeated = repmat(key_ascii, 1, ceil(length(text_ascii) / length(key_ascii)));

    % XOR operation
    encrypted_ascii = bitxor(text_ascii, key_repeated(1:length(text_ascii)));

    % ASCII values of encrypted characters
    disp('Encrypted ASCII Values:');
    disp(encrypted_ascii);

    % Converting back to characters
    encrypted_text = char(encrypted_ascii);
end
end

function streamCipher()
    % Implement Custom Stream Cipher functionality################################
    % Custom stream cipher
plaintext = 'Custom Stream Cipher Implementation';
key = 'customkey';
encrypted_text_custom = custom_stream_cipher(plaintext, key);
disp('Original Text:');
disp(plaintext);
disp('Encrypted Text (Custom Stream Cipher):');
disp(encrypted_text_custom);

function encrypted_text = custom_stream_cipher(plaintext, key)
    % Converting text and key to ASCII values
    text_ascii = double(plaintext);
    key_ascii = double(key);
    
    % Key is as long as the plaintext
    key_repeated = repmat(key_ascii, 1, ceil(length(text_ascii) / length(key_ascii)));

    % Keystream based on the key
    keystream = generate_keystream(key_repeated(1:length(text_ascii)));
    
    % XOR operation
    encrypted_ascii = bitxor(text_ascii, keystream);

    % Converting back to characters
    encrypted_text = char(encrypted_ascii);
end

function keystream = generate_keystream(key)
    % Key scheduling algorithm 
    key_length = length(key);
    keystream = zeros(1, key_length);
    
    for i = 1:key_length
        keystream(i) = bitshift(key(i), mod(i, 8));
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hashingRoutine()
    % Implement Hashing Routine functionality################################
    % Message for the hashing routine
message = 'Lets Hashing Together';
hashValue = simpleHash(message);
disp(['Hash Value for Message: ', num2str(hashValue)]);
function hashValue = simpleHash(message)
    % Simple hashing routine
    hashValue = sum(double(message));
end
end

function crcRoutine()
    % Implement CRC Routine functionality################################
    % CRC routine with a message
message = 'Check this message with CRC';
crcValue = simpleCRC(uint8(message));
disp(['CRC Value for Message: ', num2str(crcValue)]);

% Example for the CRC routine with a file
filePath = '/MATLAB Drive/file2.txt';
fileData = fileread(filePath);
crcValue = simpleCRC(uint8(fileData));
disp(['CRC Value for File: ', num2str(crcValue)]);
function crcValue = simpleCRC(data)
    % Simple CRC routine for messages or files

    % CRC polynomial coefficients (CRC-32)
    polynomial = uint32(hex2dec('EDB88320'));

    % Initialize CRC value
    crcValue = uint32(4294967295);  

    % Process each byte in the data
    for byte = data
        crcValue = bitxor(bitshift(crcValue, -8), polynomial);
        crcValue = bitxor(crcValue, uint32(byte));
        
        for bitIndex = 1:8  % Fixed the variable name
            if bitget(crcValue, 1)
                crcValue = bitxor(bitshift(crcValue, -1), polynomial);
            else
                crcValue = bitshift(crcValue, -1);
            end
        end
    end

    crcValue = bitxor(crcValue, uint32(4294967295));
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function hmacHashing()
    % Implement HMAC Hashing functionality##################################
    result = HMAC('secretkey', 'this is a very secret message', 'SHA-256');
disp(result);

function hmac_result = HMAC(key, message, method)
    % Checking if DataHash.m is in the path
    if ~exist('DataHash', 'file')
        error('DataHash.m not found. Download it from: http://www.mathworks.com/matlabcentral/fileexchange/31272-datahash');
    end
    
    % Validating the method
    valid_methods = {'SHA-1', 'SHA-256', 'SHA-384', 'SHA-512'};
    if ~ismember(method, valid_methods)
        error('Invalid hash method. Supported methods: SHA-1, SHA-256, SHA-384, SHA-512');
    end
    
    % Computing the HMAC
    block_size = 64;  % Block size for SHA-1 and SHA-256
    if strcmp(method, 'SHA-384') || strcmp(method, 'SHA-512')
        block_size = 128;  % Block size for SHA-384 and SHA-512
    end
    
    key = HMAC_prepare_key(key, block_size);
    
    ipad = char(ones(1, block_size) * hex2dec('36'));
    opad = char(ones(1, block_size) * hex2dec('5C'));
    
key_xor_ipad = char(bitxor(uint8(key), uint8(ipad)));
key_xor_opad = char(bitxor(uint8(key), uint8(opad)));
    
    inner_hash_input = [key_xor_ipad, message];
    inner_hash_output = DataHash(inner_hash_input, 'OutputAs', 'char', 'SHA-256');
    
    hmac_result = DataHash([key_xor_opad, inner_hash_output], 'OutputAs', 'char', 'SHA-256');
    
    hmac_result = upper(hmac_result);
end

function key = HMAC_prepare_key(key, block_size)
    % If the key longer than block size, hashing it
    if length(key) > block_size
        key = DataHash(key, 'OutputAs', 'char', 'SHA-256');
    end
    
    % If the key shorter than block size, padding with zeros
    if length(key) < block_size
        key = [key, char(zeros(1, block_size - length(key)))];
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gcmBlockCipher()
    % Implement GCM Block Cipher functionality
    plaintext = 'Authenticated encryption algorithms';
shift = 3;
encryptedText = encryptCaesar(plaintext, shift);
disp(['Encrypted Text: ' encryptedText]);

decryptedText = decryptCaesar(encryptedText, shift);
disp(['Decrypted Text: ' decryptedText]);

function encryptedText = encryptCaesar(plainText, shift)
    plainText = upper(plainText);
    
    encryptedText = '';
    
    for i = 1:length(plainText)
        if isletter(plainText(i))
            encryptedChar = char(mod(double(plainText(i)) - 65 + shift, 26) + 65);
        else
            encryptedChar = plainText(i);
        end
        
        encryptedText = [encryptedText, encryptedChar];
    end
end

function decryptedText = decryptCaesar(encryptedText, shift)
    decryptedText = encryptCaesar(encryptedText, -shift);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 9
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function growthRatesVisualization()
    % Implement Growth Rates Visualization functionality##################################
    % Visualization of Growth Rates
% Generating input values (n)
n = 1:20;

% Plotting constant, linear, logarithmic, quadratic, and exponential growth
figure;
plot(n, ones(size(n)), '-o', 'DisplayName', 'O(1) - Constant');
hold on;
plot(n, n, '-o', 'DisplayName', 'O(n) - Linear');
plot(n, log2(n), '-o', 'DisplayName', 'O(log n) - Logarithmic');
plot(n, n.^2, '-o', 'DisplayName', 'O(n^2) - Quadratic');
plot(n, 2.^n, '-o', 'DisplayName', 'O(2^n) - Exponential');

% Setting axis labels and title
xlabel('Input Size (n)');
ylabel('Operations');
title('Complexity Growth Rates');
legend('Location', 'northwest');

% Timing Mechanisms

% Using tic and toc
tic;
constantTime();
elapsedTime = toc;
disp(['Elapsed Time: ', num2str(elapsedTime), ' seconds']);

% Using timeit function
elapsedTime = timeit(@() linearTime(100));
disp(['Elapsed Time: ', num2str(elapsedTime), ' seconds']);

% Using profile
profile on;
quadraticTime(50);
profile off;

% Checking if profview is supported
if ~isdeployed
    try
        profview;
    catch
        disp('profview is not supported on this platform.');
    end
else
    disp('profview is not supported on this platform.');
end

% Importance of Complexity in Cryptography
disp('The measure of complexity is important in cryptography because:');
disp('- Cryptographic algorithms need to be efficient to handle large data sets.');
disp('- Time complexity affects the practicality of the algorithm in real-world applications.');
disp('- A balance between security and performance is crucial for cryptographic algorithms.');


% Stats Option in the Menu
disp('You can implement a stats option in the menu to provide information on timings, memory usage, etc.');


% Time Complexity Demonstrations

% Constant time complexity - O(1)
function constantTime()
    disp('Constant Time Complexity - O(1)');
end

% Linear time complexity - O(n)
function linearTime(n)
    disp(['Linear Time Complexity - O(n), n = ', num2str(n)]);
    for i = 1:n
        
    end
end

% Logarithmic time complexity - O(log n)
function logarithmicTime(n)
    disp(['Logarithmic Time Complexity - O(log n), n = ', num2str(n)]);
    for i = 1:log2(n)
        
    end
end

% Quadratic time complexity - O(n^2)
function quadraticTime(n)
    disp(['Quadratic Time Complexity - O(n^2), n = ', num2str(n)]);
    for i = 1:n
        for j = 1:n
            
        end
    end
end

% Exponential time complexity - O(2^n)
function exponentialTime(n)
    disp(['Exponential Time Complexity - O(2^n), n = ', num2str(n)]);
    for i = 1:2^n
       
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  week 10
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rsaAlgorithm()
    % Implement RSA Algorithm functionality##################################
    % Testing the RSA algorithm
originalMessage = 'Testing the RSA algorithm!';
disp('Original Message:');
disp(originalMessage);

% Key generation
[publicKey, privateKey] = generateRSAKeys();

% Encryption
encryptedMessage = encryptRSA(originalMessage, publicKey);
disp('Encrypted Message:');
disp(encryptedMessage);

% Decryption
decryptedMessage = decryptRSA(encryptedMessage, privateKey);
disp('Decrypted Message:');
disp(decryptedMessage);

% RSA Algorithm Implementation
% Key Generation
function [publicKey, privateKey] = generateRSAKeys()
    % Choosing two large prime numbers
    p = 61;
    q = 53;
    % Computing modulus
    n = p * q;
    % Computing totient
    totient = (p - 1) * (q - 1);
    % Choosing public exponent (e) such that 1 < e < totient and gcd(e, totient) = 1
    e = 17;
    % Computing private exponent (d) such that d*e mod totient = 1
    d = modinv(e, totient);

    % Public Key: (e, n)
    publicKey.e = e;
    publicKey.n = n;

    % Private Key: (d, n)
    privateKey.d = d;
    privateKey.n = n;
end

% Encryption
function encryptedMessage = encryptRSA(originalMessage, publicKey)
    % Converting the message to integers
    messageIntegers = uint64(originalMessage);

    % Encrypting each integer separately
    encryptedIntegers = powermod(messageIntegers, publicKey.e, publicKey.n);

    % Converting the encrypted integers back to characters
    encryptedMessage = char(encryptedIntegers);
end

% Decryption
function decryptedMessage = decryptRSA(encryptedMessage, privateKey)
    % Converting the encrypted message to integers
    encryptedIntegers = uint64(encryptedMessage);

    % Decrypting each integer separately
    decryptedIntegers = powermod(encryptedIntegers, privateKey.d, privateKey.n);

    % Converting the decrypted integers back to characters
    decryptedMessage = char(decryptedIntegers);
end

% Function for modular inverse
function inv = modinv(a, m)
    % Extended Euclidean Algorithm
    [g, x, ~] = gcd(a, m);
    
    if g == 1
        % Ensuring x is positive
        inv = mod(x, m);
    else
        error('The modular inverse does not exist.');
    end
end

function result = powermod(base, exponent, modulus)
    % Calculating (base^exponent) mod modulus efficiently
    result = ones(size(base), 'uint64'); % Initializing result with ones
    base = mod(base, modulus); % Fixing for non-scalar base

    while exponent > 0
        if bitget(exponent, 1) == 1
            result = mod(result .* base, modulus);
        end
        base = mod(base.^2, modulus);
        exponent = bitshift(exponent, -1);
    end
end
end
