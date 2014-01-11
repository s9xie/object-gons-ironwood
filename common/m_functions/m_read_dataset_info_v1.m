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
%-------------------------------------------------
%v0 -> v1 support all datasets and support read bbox files
%-------------------------------------------------
function return_state = m_read_dataset_info_v1(global_config)

return_state = 1; % valid

switch global_config.dataset
    case 'HMDB51'
        vid_paths = {};
        vid_names = {};
        vid_bbox_names = {};
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
                [~, vid_name_noext, ~] = fileparts(file_names(j).name);
                vid_bbox_names{i_class}{i_vid} = [vid_name_noext,'.bb'];
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
        
        save(global_config.read_dataset_info.file_name,'vid_paths','vid_names','vid_bbox_names','class_names','vid_nums_in_class','class_num','vid_total_num',...
            'splits_num','test_set_idx','train_set_idx');
        
        
    case 'Hollywood2'
        
        
        class_names = {'AnswerPhone','DriveCar','Eat','FightPerson','GetOutCar','HandShake','HugPerson','Kiss','Run','SitDown','SitUp','StandUp'};
        class_num = length(class_names);
        
        
        vid_nums_in_class = [];
        
        % read the txt
        train_set_label = cell(1,class_num);
        test_set_label = cell(1,class_num);
        
        vid_paths = cell(class_num,1);
        vid_names = cell(class_num,1);
        
        splits_num = 1;
        i_split = 1;
        train_set_idx = {};
        test_set_idx = {};
        
        train_set_gl_idx = {};
        test_set_gl_idx = {};
        
        for i_class = 1:class_num
            
            i_vid = 0;
            i_gl_vid = 0;
            
            cur_train_set_labels = [];
            cur_train_set_idx = [];
            cur_train_set_gl_idx = [];
            
            fname = fullfile(global_config.dataset_path,'ClipSets',[class_names{i_class},'_train.txt']);
            fid = fopen(fname);
            while 1
                tline = fgetl(fid);
                if tline==-1
                    break
                end
                i_gl_vid = i_gl_vid +1;
                
                [tline, u] = strtok(tline,' ');
                u = str2num(u);
                
                cur_train_set_labels = [cur_train_set_labels;u];
                
                if u == 1
                    i_vid = i_vid +1;
                    
                    cur_train_set_idx = [cur_train_set_idx;i_vid];
                    cur_train_set_gl_idx = [cur_train_set_gl_idx;i_gl_vid];
                    video_name = fullfile('AVIClips',[tline,'.avi']);
                    vid_paths{i_class}{i_vid} = video_name;
                    vid_names{i_class}{i_vid} = [tline,'.avi'];
                end
            end
            
            
            
            fclose(fid);
            
            train_set_label{i_class} = cur_train_set_labels;
            train_set_idx{i_split}{i_class} = cur_train_set_idx;
            train_set_gl_idx{i_split}{i_class} = cur_train_set_gl_idx;
            
            
            cur_test_set_labels = [];
            cur_test_set_idx = [];
            cur_test_set_gl_idx = [];
            
            fname = fullfile(global_config.dataset_path,'ClipSets',[class_names{i_class},'_test.txt']);
            
            
            i_gl_vid = 0;
            fid = fopen(fname);
            while 1
                tline = fgetl(fid);
                if tline==-1
                    break
                end
                i_gl_vid = i_gl_vid +1;
                
                [tline, u] = strtok(tline,' ');
                u = str2num(u);
                
                cur_test_set_labels = [cur_test_set_labels;u];
                
                if u == 1
                    i_vid = i_vid +1;
                    
                    cur_test_set_idx = [cur_test_set_idx;i_vid];
                    cur_test_set_gl_idx = [cur_test_set_gl_idx;i_gl_vid];
                    video_name = fullfile('AVIClips',[tline,'.avi']);
                    vid_paths{i_class}{i_vid} = video_name;
                    vid_names{i_class}{i_vid} = [tline,'.avi'];
                    vid_bbox_names{i_class}{i_vid} = fullfile(global_config.dataset_path,[tline,'.bb']);
                end
            end
            
            fclose(fid);
            
            test_set_label{i_class} = cur_test_set_labels;
            test_set_idx{i_split}{i_class} = cur_test_set_idx;
            test_set_gl_idx{i_split}{i_class} = cur_test_set_gl_idx;
            
            train_num_per_class(i_class) = length(cur_train_set_idx);
            test_num_per_class(i_class) = length(cur_test_set_idx);
            
            vid_nums_in_class = [vid_nums_in_class,i_vid];
        end
        
        % add another info on the multi_labeled ones
        % train_valid{i_class}{j_class} is the valid label of the ith class when the cur pos class is jth class
        train_valid = {};
        test_valid = {};
        i_split = 1;
        for j_class = 1:class_num
            
            cur_label_train =  train_set_label{j_class};
            cur_label_test =  test_set_label{j_class};
            
            train_valid{j_class}{j_class} = cur_label_train(train_set_gl_idx{i_split}{j_class}) == 1;
            test_valid{j_class}{j_class} = cur_label_test(test_set_gl_idx{i_split}{j_class}) == 1;
            
            for i_class = 1:class_num
                if j_class == i_class
                    continue;
                else
                    train_valid{i_class}{j_class} = cur_label_train(train_set_gl_idx{i_split}{i_class}) == -1;
                    test_valid{i_class}{j_class} = cur_label_test(test_set_gl_idx{i_split}{i_class}) == -1;
                    
                    cur_label_train(train_set_gl_idx{i_split}{i_class}) = 0;
                    cur_label_test(test_set_gl_idx{i_split}{i_class}) = 0;
                end
                
            end
            
        end
        
        vid_total_num = length(test_set_label{1}) + length(train_set_label{1});
        
        % debug use
        %         for j_class = 1:class_num
        %             train_tot_valid = 0;
        %             test_tot_valid = 0;
        %
        %             for i_class = 1:class_num
        %                 train_tot_valid = train_tot_valid + sum(train_valid{i_class}{j_class});
        %                 test_tot_valid = test_tot_valid + sum(test_valid{i_class}{j_class});
        %             end
        %
        %             train_tot_valid
        %             test_tot_valid
        %         end
        
        %       vid_total_num = ;
        %       vid_nums_in_class
        %       test_set_idx
        %       train_set_idx
        
        
        save(global_config.read_dataset_info.file_name,'vid_paths','vid_names','vid_bbox_names','class_names','vid_nums_in_class','class_num','vid_total_num',...
            'splits_num','test_set_idx','train_set_idx','train_valid','test_valid');
        
        
    case 'UCF11'
        
        vid_paths = {};
        vid_names = {};
        vid_bbox_names = {};
        class_names = {};
        vid_nums_in_class = [];
        
        data_splits = {};
        
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
            
            
            sub_class_folder_names = dir(fullfile(global_config.dataset_path,i_class_names));
            i_vid = 0;
            
            i_sub_folder = 0;
            for k = 1:length(sub_class_folder_names)
                
                if strcmp(sub_class_folder_names(k).name,'.') || strcmp(sub_class_folder_names(k).name,'..') || strcmp(sub_class_folder_names(k).name,'Annotation')
                    continue;
                end
                i_sub_folder = i_sub_folder +1;
                data_splits{i_sub_folder}{i_class} = [];
                file_names = dir(fullfile(global_config.dataset_path,i_class_names,sub_class_folder_names(k).name));
                
                for j = 1:length(file_names)
                    if strcmp(file_names(j).name,'.') || strcmp(file_names(j).name,'..')
                        continue;
                    end
                    i_vid = i_vid +1;
                    
                    data_splits{i_sub_folder}{i_class} = [data_splits{i_sub_folder}{i_class};i_vid];
                    
                    vid_total_num = vid_total_num + 1;
                    vid_paths{i_class}{i_vid} = fullfile(i_class_names,sub_class_folder_names(k).name,file_names(j).name);
                    vid_names{i_class}{i_vid} = file_names(j).name;
                    [~, vid_name_noext, ~] = fileparts(file_names(j).name);
                    vid_bbox_names{i_class}{i_vid} = fullfile(global_config.dataset_path,[vid_name_noext,'.bb']);
                end
                               
                
                
            end
            
            vid_nums_in_class = [vid_nums_in_class;i_vid];
            
            
        end
        
        if (i_class == 0)
            fprintf('no classed exsit!\n');
            return_state = 0;
            return;
        end
        
        class_num = i_class;
        
        % read the split info
        
        splits_num = length(data_splits);
        
        train_set_idx = {}; %1*nClass cells,each cell store the train idx in the related class
        test_set_idx = {};  %1*nClass cells,each cell store the test idx in the related class
        
        for i_split = 1:splits_num
            
            for i_class = 1:class_num
                
                train_set_idx{i_split}{i_class} = [];
                test_set_idx{i_split}{i_class} = [];
                
                for i_data_split = 1:splits_num
                    if i_data_split == i_split
                        test_set_idx{i_split}{i_class} = [test_set_idx{i_split}{i_class};data_splits{i_data_split}{i_class}];
                    else
                        train_set_idx{i_split}{i_class} = [train_set_idx{i_split}{i_class};data_splits{i_data_split}{i_class}];
                    end
                end
                
            end % i_class ends
            
        end % i_splits ends
       
        % save
        
        save(global_config.read_dataset_info.file_name,'vid_paths','vid_names','class_names','vid_nums_in_class','class_num','vid_total_num',...
            'splits_num','test_set_idx','train_set_idx');
        
        
    otherwise
        fprintf('WRONG!!! unknown dataset!\n');
        return_state = 0;
        return;
        
        
end

end