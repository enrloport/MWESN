function mean_error_A(A)
    return mean(map(v -> v[2] , values(A)))
end