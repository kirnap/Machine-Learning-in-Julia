# Implements the selection sort algorithm
using Debug


function swap2!(s::Array, i::Integer, k::Integer)
  temp = s[i]
  s[i] = s[k]
  s[k] = temp
end


function omerselect(s::Array)
	for i=1:length(s)
		minIndex = i
		for k=i+1:length(s)
	  		(s[minIndex] > s[k]) && (minIndex = k)
		end
		(i!=minIndex) && (swap2!(s, minIndex, i))
	end
	return s
end

function test()
  a = [8, 4, 2, 5, 42]
  b = rand(4,1)
  println(omerselect(a))

  println("=======")
  omerselect(b)
end
test()
