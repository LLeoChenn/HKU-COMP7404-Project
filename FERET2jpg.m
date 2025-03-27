image_height = 64;
image_width = 64;
save_folder = 'D:\groupwork_data\FERET_4096'; 

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

train_folder = fullfile(save_folder, 'train');
test_folder = fullfile(save_folder, 'test');
if ~exist(train_folder, 'dir')
    mkdir(train_folder);
end
if ~exist(test_folder, 'dir')
    mkdir(test_folder);
end

% Obtain unique person labels
unique_labels = unique(gnd);
num_labels = length(unique_labels);

% Calculate the number of persons to be allocated to the training set
num_train_labels = round(num_labels * 0.7);

% Randomly select 70% of the person labels as the training set
shuffled_labels = unique_labels(randperm(num_labels));
train_labels = shuffled_labels(1:num_train_labels);
test_labels = shuffled_labels(num_train_labels + 1:end);

% Used to record the number of images in each category
train_class_counts = zeros(max(gnd), 1);
test_class_counts = zeros(max(gnd), 1);

% Iterate through all images
num_images = size(fea, 1);
for i = 1:num_images
    % Extract the one - dimensional vector of the i - th image
    image_vector = fea(i, :);
    
    % Reshape the one - dimensional vector into a two - dimensional matrix
    image_matrix = reshape(image_vector, image_height, image_width);
    
    % Normalize the data to the range [0, 1]
    image_matrix = (image_matrix - min(image_matrix(:))) / (max(image_matrix(:)) - min(image_matrix(:)));
    
    % Convert the data type to uint8 (range [0, 255])
    image_matrix = uint8(image_matrix * 255);
    
    % Get the label of the current image
    label = gnd(i);
    
    if ismember(label, train_labels)
        % This image belongs to the training set
        % Increment the number of images in the corresponding category
        train_class_counts(label) = train_class_counts(label) + 1;
        
        % Generate the image number within this category
        image_num_in_class = train_class_counts(label);
        
        % Generate the path of the sub - folder for this category
        class_folder = fullfile(train_folder, sprintf('class_%d', label));
        
        % Create the sub - folder for this category if it does not exist
        if ~exist(class_folder, 'dir')
            mkdir(class_folder);
        end
        
        % Generate the full file path
        filename = fullfile(class_folder, sprintf('image_%d.jpg', image_num_in_class));
        
    else
        % This image belongs to the test set
        % Increment the number of images in the corresponding category
        test_class_counts(label) = test_class_counts(label) + 1;
        
        % Generate the image number within this category
        image_num_in_class = test_class_counts(label);
        
        % Generate the path of the sub - folder for this category
        class_folder = fullfile(test_folder, sprintf('class_%d', label));
        
        % Create the sub - folder for this category if it does not exist
        if ~exist(class_folder, 'dir')
            mkdir(class_folder);
        end
        
        % Generate the full file path
        filename = fullfile(class_folder, sprintf('image_%d.jpg', image_num_in_class));
    end
    
    % Save the image as a JPG file
    imwrite(image_matrix, filename);
end