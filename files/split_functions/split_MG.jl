function split_data_mg(data, initial_transient, train_length, test_length)

    d,it,trl,tel = data, initial_transient, train_length, test_length

    train_data    = d[1:trl]
    train_labels  = d[it+2:trl+1]
    test_data     = d[trl+1:trl+tel]
    test_labels   = d[trl+2:trl+tel+1]
    
    return train_data, train_labels, test_data, test_labels
end