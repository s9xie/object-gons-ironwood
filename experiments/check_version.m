% check the version of every parts, is needed to update, the version +1, and do the part
% wyw @MSRA @9/5/2013

function global_config_checked = check_version(global_config)
    global_config_checked = global_config;
    global_config_checked.valid = true;
    
    %% check the dataset
    if ~exist(global_config_checked.dataset_path,'dir') 
        fprintf('the dataset path is not exist! %s\n',global_config_checked.dataset_path);
        global_config_checked.valid = false;
        return;
    end
    
    %%  check local path
    if ~exist(global_config_checked.local_path,'dir')
        fprintf('the local path is not exist! %s generate the path\n',global_config_checked.dataset_path);
        mkdir(global_config_checked.local_path);
    end
    
    %% check the dataset_info.mat
    if ~exist(global_config.read_dataset_info.file_name,'file')
        fprintf('[%s] not exist! will generate it!\n',global_config.read_dataset_info.file_name);
        global_config.read_dataset_info.do = 1;
    else
        fprintf('[%s] already exist!\n',global_config.read_dataset_info.file_name);
        if global_config.read_dataset_info.do == 1
            global_config_checked.read_dataset_info.do = 0;
        end
    end
    
    %% extract_features part
    
    dir_path = global_config.extract_features.path;
    if ~exist(dir_path,'dir') && global_config.extract_features.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.extract_features.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.extract_features.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    %% learn_codebooks part
    dir_path = global_config.learn_codebooks.path;
    if ~exist(dir_path,'dir') && global_config.learn_codebooks.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.learn_codebooks.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.learn_codebooks.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    %% coding part
    
    dir_path = global_config.coding.path;
    if ~exist(dir_path,'dir') && global_config.coding.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.coding.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.coding.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    %% extract_low_level_descriptors
    
    dir_path = global_config.extract_low_level_descriptors.path;
    if ~exist(dir_path,'dir') && global_config.extract_low_level_descriptors.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.extract_low_level_descriptors.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.extract_low_level_descriptors.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    %% clustering
    
    dir_path = global_config.clustering.path;
    if ~exist(dir_path,'dir') && global_config.clustering.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.clustering.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.clustering.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    
    %% extract_part_features
    
    dir_path = global_config.extract_part_features.path;
    if ~exist(dir_path,'dir') && global_config.extract_part_features.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.extract_part_features.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.extract_part_features.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    
    %% learn_action_parts
    
    dir_path = global_config.learn_action_parts.path;
    if ~exist(dir_path,'dir') && global_config.learn_action_parts.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.learn_action_parts.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.learn_action_parts.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    
    
    %% extract_mid_level_descriptors
    
    dir_path = global_config.extract_mid_level_descriptors.path;
    if ~exist(dir_path,'dir') && global_config.extract_mid_level_descriptors.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.extract_mid_level_descriptors.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.extract_mid_level_descriptors.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    
    %% classification
    
    dir_path = global_config.classification.path;
    if ~exist(dir_path,'dir') && global_config.classification.do
        fprintf('[%s] not exist! will generate it!\n',dir_path);
        mkdir(dir_path);
        global_config.classification.do = 1;
    else
        fprintf('[%s] already exist!\n',dir_path);
    end
    
    % update the global 
    if global_config.classification.do
        save(fullfile(dir_path,'global_config.mat'),'global_config');
    end
    
    
end