#=
To provide some helping functions which are more MATLAB like style
=#


function modified_reshape(matrix; row_number=0, column_number=0)
  #=
  This function allows you not to specify all the dimension of a reshape matrix
  Simply write the dimension that you want to change and let the remaining part
  is done by function itself
  =#
  row, column = size(matrix)
  number_of_entries = row * column

  if row_number == 0 && column_number == 0
    return matrix

  elseif row_number != 0 && column_number == 0
    if number_of_entries % row_number != 0
      throw(DimensionMismatch("Dimensions must be consistent with input matrix"))
    end
    return reshape(matrix, row_number, convert(Int64, number_of_entries / row_number))

  elseif row_number == 0 && column_number != 0
    if number_of_entries % column_number != 0
      throw(DimensionMismatch("Dimensions must be consistent with input matrix"))
    end
    return reshape(matrix, convert(Int64, number_of_entries / column_number), column_number)

  else
    if row_number * column_number != number_of_entries
      throw(DimensionMismatch("Dimensions must be consistent with input matrix"))
    end
      return reshape(matrix, row_number, column_number)
  end
end
