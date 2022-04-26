function [meas_data, outlier_idx] = rm_meas_data_outlier(meas_data,FLAG)
if(FLAG)
    meas_data2 = meas_data - mean(meas_data(:));
    meas_data2 = meas_data2./max(meas_data2(:));
    meas_data2 = meas_data2(:);
    outlier_idx = isoutlier(meas_data2);
    meas_data(outlier_idx) = 0;
else
    outlier_idx = logical(zeros(size(meas_data)));
end

meas_data = meas_data(:);
meas_data(~outlier_idx) = meas_data(~outlier_idx) - mean(meas_data(~outlier_idx));
meas_data = meas_data./max(meas_data);
end