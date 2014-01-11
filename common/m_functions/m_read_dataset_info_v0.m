% read the dataset video class and store the dataset info
% output:
% dataset_info.mat  @root_path\local\[dataset_name]\
%        vid_path: reference to the [root_path\datasets\] stored in cell vid_paths{i_class}{i_file}
%        class_names: the name of each class stored in cell vid_names{i_class}
%        vid_names: the name of each video stored in cell vid_names{i_class}
%        class_num: class number
%        vid_total_num: total videos number
%        vid_nums_in_class: matrix of class_num x 1
%        splits_num: splits of the data set
%        train_set_idx:  train_set_idx{i_splits}{i_class} store the  train_set in i_class of i_splits
%        test_set_idx:   test_set_idx{i_splits}{i_class}

% wyw @MSRA @9/6/2013
%
% v1: support read bbox file
function return_state = m_read_dataset_info_v0(global_config)

return_state = 1; % valid 

switch global_config.dataset
    case 'HMDB51'
        vid_paths = {};
        vid_names = {};
        class_names = {};
        vid_nums_in_class = [];
        
        class_folder_names = dir(global_config.dataset_path);
        i_class = 0;
        vid_total_num = 0;
        for i = 1:length(class_folder_names)
            
            if strcmp(class_folder_names(i).name,'.') || strcmp(class_folder_names(i).name,'..')
                continue;
            end
            
            i_class = i_class+1;
            i_class_names = class_folder_names(i).name;
            class_names{i_class} = i_class_names;
            
            file_names = dir(fullfile(global_config.dataset_path,i_class_names));
            
            i_vid = 0;
            for j = 1:length(file_names)
                if strcmp(file_names(j).name,'.') || strcmp(file_names(j).name,'..')
                    continue;
                end
                i_vid = i_vid +1;
                vid_total_num = vid_total_num + 1;
                vid_paths{i_class}{i_vid} = fullfile(i_class_names,file_names(j).name);
                vid_names{i_class}{i_vid} = file_names(j).name;
            end
            
            vid_nums_in_class = [vid_nums_in_class;i_vid];
            
        end
        
        if (i_class == 0)
            fprintf('no classed exsit!\n');
            return_state = 0;
            return;
        end
        
        class_num = i_class;
        
        train_num_per_class = 70;
        test_num_per_class = 30;
        % read the split info
        if ~exist(global_config.splits_file_path,'dir')
            fprintf('no splits files exist for dataset HMDB51!\n');
            return_state = 0;
            return;
        else
            splits_num = 3;
            
            train_set_idx = {}; %1*nClass cells,each cell store the train idx in the related class
            test_set_idx = {};  %1*nClass cells,each cell store the test idx in the related class
                
            for i_split = 1:splits_num
                
               
                [train_data_names,test_data_names] = get_HMDB_split(i_split,global_config.splits_file_path);
                
                for i_class = 1:class_num
                    
                    cur_train_idx = [];
                    for i_tr = 1:length(train_data_names{i_class})
                        for idx = 1:length(vid_names{i_class})
                            if strcmp(train_data_names{i_class}{i_tr},vid_names{i_class}{idx})
                                cur_train_idx = [cur_train_idx;idx];
                                break;
                            end
                        end
                    end
                    train_set_idx{i_split}{i_class} = cur_train_idx;
                    if (train_num_per_class ~= length(cur_train_idx))                    
                        fprintf('get train data set split wrong at class%03d!\n',i_class);
                        return_state = 0;
                        return;
                    end
                    cur_test_idx = [];
                    for iTe = 1:length(test_data_names{i_class})
                        for idx = 1:length(vid_names{i_class})
                            if strcmp(test_data_names{i_class}{iTe},vid_names{i_class}{idx})
                                cur_test_idx = [cur_test_idx;idx];
                                break;
                            end
                        end
                    end
                    test_set_idx{i_split}{i_class} = cur_test_idx;
                    if (test_num_per_class ~= length(cur_test_idx))
                        fprintf('get test data set split wrong at class%03d!\n',i_class);
                        return_state = 0;
                        return;
                    end
                    
                end % i_class ends
                
            end % i_splits ends
        
        end % is split file exist
        
        % save 
        
        save(global_config.read_dataset_info.file_name,'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num',...
            'splits_num','test_set_idx','train_set_idx','test_num_per_class','train_num_per_class');
        
    otherwise
        fprintf('unknown dataset!');
        return;
        
        
end

end