# The department abstract and authornames were acquired from https://lup.lub.lu.se/search/publication?q=department+exact+v1000253 and exported as an excel file [publications.xsl](https://github.com/baggepinnen/MatchScholars.jl/blob/master/publications.xsl).

using LinearAlgebra, Statistics
using TextAnalysis, ExcelReaders, StringDistances, AMD, SparseArrays, Plots, Latexify
cd(@__DIR__)


# # Read department data
filename  = "publications.xls"
authors   = readxl(filename, "Sheet1!E2:E1501")[:]
abstracts = readxl(filename, "Sheet1!X2:X1501")[:];

# Filter the data to only keep valid names and abstracts
valid_authors   = isa.(authors,String)
authors         = authors[valid_authors]
abstracts       = abstracts[valid_authors]

valid_abstracts = isa.(abstracts,String)
authors         = authors[valid_abstracts]
abstracts       = abstracts[valid_abstracts]

# Some abstracts are in Swedish, we get rid of those
valid_abstracts = [match(r"ENG", a) != nothing && match(r"och", a) == nothing for a in abstracts]
authors         = authors[valid_abstracts]
abstracts       = abstracts[valid_abstracts]

# Clean data
authors   = [replace(a, r"\([\w-]+\)" => "") for a in authors]
abstracts = [replace(a, r"\([\w-]+\)" => "") for a in abstracts]
abstracts = [replace(a, r"ENG:"       => "") for a in abstracts]
@assert length(authors) == length(abstracts)



# Prepare data for text analysis
docs = StringDocument.(deepcopy(abstracts))
crps = Corpus(deepcopy(docs))
prepare!(crps, strip_corrupt_utf8 | strip_case | strip_articles | strip_prepositions | strip_pronouns | strip_stopwords | strip_whitespace | strip_non_letters | strip_numbers)
remove_words!(crps, ["br", "control", "system", "systems"]) # For some reason the word "br" appeas very often (html tag?)
update_lexicon!(crps)

# # Analysis
# We will perform [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
# Ideally, since we are going to use the founds topics for similarity analysis, we should use a correlated topic model (CTM). I could not find a working implementation of that and didn't have the time to fix one, so LDA will have to do. We can estimate topic correlations adhoc using either ϕ*ϕ' (on a word similarity basis) or θ*θ' (on author similarity basis)

# LDA
m     = DocumentTermMatrix(crps)
k     = 6    # number of topics
iters = 1000 # number of gibbs sampling iters
α     = 1/k  # hyper parameter topics per document
β     = 0.01 # hyper parameter words per topic
ϕ,θ   = lda(m, k, iters, α, β) # ϕ: topics × words θ: topics × documents
println("Occupancy: ", sum(ϕ.!=0)/length(ϕ))

# Calculate a topic matrix if you want to inspect the found topics
k_largest(array,k) = sortperm(array, rev=true)[1:k]
words_per_topic = 20
topics = map(1:k) do topic_num
    probs = Vector(ϕ[topic_num,:])
    inds_of_largest = k_largest(probs,words_per_topic)
    words = m.terms[inds_of_largest]
    words
end
topics = hcat(topics...)
# `topics` is now a matrix where each column consists of the 20 most prominent words in each topic


# We can also define some interesting covaiance matrices for visualization (plots omitted)
function corr(x::AbstractMatrix)
    d = diag(x)
    y = copy(x)
    for i = 1:length(d), j = 1:length(d)
        y[i,j] /= sqrt(x[i,i]*x[j,j])
    end
    y
end
topic_covariance_by_words      = Matrix(ϕ*ϕ')
topic_covariance_by_documents  = Matrix(θ*θ')
topic_correlation_by_words     = corr(topic_covariance_by_words)
topic_correlation_by_documents = corr(topic_covariance_by_documents)
document_covariance            = θ'θ

# We now conduct some tedious coding to associate the authors of the abstracts in the publication database with the current staff members
individual_authors = String.(strip.(vcat(split.(authors, ";")...)))
unique_authors     = unique(individual_authors)
author_mapping = map(unique_authors) do ua
    [match(Regex(ua), aa) != nothing for aa in authors]
end
author_mapping = findall.(author_mapping)
author_rank    = sortperm(length.(author_mapping), rev=true)
top_10_authors = unique_authors[author_rank[1:10]]

# The current staff at the department is listed in [staff.txt](https://github.com/baggepinnen/MatchScholars.jl/blob/master/staff.txt)
# Since the names of the authors are not on the same format in the database and on the webpage, we use a [string comparison tool](https://github.com/matthieugomez/StringDistances.jl) to find correspondences.
# "The distance Tokenmax(RatcliffObershelp()) is a good choice to link names or adresses across datasets."
staff = readlines("staff.txt")

staff2unique_author = map(staff) do staffi
    distances = map(unique_authors) do authori
        StringDistances.compare(TokenMax(RatcliffObershelp()), filter(isascii,authori), filter(isascii,staffi))
    end
    max_i = argmax(distances)
    distances[max_i] < 0.9 && return 0
    return max_i
end
filter!(!iszero, staff2unique_author)
authornames = unique_authors[staff2unique_author]

document_indices_of_staff = map(authornames) do staffi
    r = Regex(staffi)
    present = map(authors) do authori
        match(r, authori) != nothing
    end
    findall(present)
end


# To find the topic vectors of each staff members, we average over all their publications. This is a simple way of doing things, but might cause a senior author with a diverse set of publications to appear as having little similarity with a young researcher with a narrow focus, even if the senior author has a few publications in that particular topic.
staff_vectors = map(document_indices_of_staff) do staffdocinds
    # ϕ: topics × words   θ: topics × documents
    mean(θ[:,staffdocinds], dims=2)
end
staff_vectors = hcat(staff_vectors...) # n_topics × n_staff

# Let's see which authors cover which topics
latexify(topics, env=:mdtable, latex=false, head=["Topic $i" for i = 1:k]) |> display
heatmap(staff_vectors',yticks=(1:length(authornames), authornames), xlabel="Topic", ylabel="Author", size=(400,1000), color=:blues)

# We can also analyze the covariance between authors
# Before we plot the covariance matrix, we try to approximately diagonalize it using the [AMD algorithm](https://github.com/JuliaSmoothOptimizers/AMD.jl). To do this, we have to set some elements that fall beneath a certain threshold to zero. We plot a histogram to assist us in setting this threshold
function diagonalize(C, tol; permute_y=false, doplot=true)
    C = copy(C)
    amdmat = size(C,1) == size(C,2) ? copy(C) : C'C
    # amdmat = C'C
    doplot && (histogram(abs.(amdmat[:])) |> display)
    amdmat[abs.(amdmat) .< tol] .= 0
    permutation = amd(sparse(amdmat))
    ypermutation = permute_y ? permutation : 1:size(C,1)
    C[ypermutation,permutation], permutation, ypermutation
end
function plotcovariance(C, xvector, yvector; kwargs...)
    xticks = (1:length(xvector), xvector)
    yticks = (1:length(yvector), yvector)
    heatmap(C; xticks=xticks, yticks=yticks, xrotation=90, title="Author similarity", kwargs...)
end


# The correlation between the abstracts of the staff are given by the inner product of their respective topic vectors
staff_covariance = staff_vectors'topic_correlation_by_documents*staff_vectors

# It seems that 0.3 is a reasonable threshold
C, permutation, ypermutation = diagonalize(staff_covariance, 0.3, permute_y=true, doplot=true)
plotcovariance(C,authornames[permutation],authornames[permutation], xrotation=90, size=(1000,1000), yflip=true)


# # Staff connections
# We can analyze the co-authorship between the staff members by counting the number of articles they have authored together
coauthor_graph = map(Iterators.product(authornames,authornames)) do (s1,s2)
    count(occursin(s1,a) && occursin(s2,a) for a in authors)
end
# We plot the `coauthor_graph` matrix has a heatmap. Since some authors have many more publications than others, we transform the data using `log(1+x)`.
ticks = (collect(eachindex(authornames)), authornames)
heatmap(log1p.(coauthor_graph), xticks=ticks, yticks=ticks, xrotation=90, title="Co-author graph (log(1+x))", size=(800,600), yflip=true)
# We see that the tightest co-authors are Anders Robertsson and Rolf Johansson. We can also figure out who is the supervisor of whom, for instance, Anders Robertsson and Rolf Johansson ar my (Fredrik Bagge Carlson) supervisors.

# We can also use some graph tools to make a fancier plot
using LightGraphs, GraphPlot, Compose, Cairo, Colors
graph = SimpleGraph(log1p.(coauthor_graph))

nodesize = diag(log.(0.1 .+ coauthor_graph))
lastname(n) = split(n,",")[1]
gplot(graph, nodelabel=lastname.(authornames), nodesize=1nodesize, nodelabelsize=0.1, layout=(args...)->spring_layout(args...; C=20))

#
using Dates
println("Compiled: ", now())
# To compile this page, run `Literate.notebook("department.jl", ".", documenter=false, execute=false, credit=false); convert_doc("department.ipynb", "department.jmd"); weave("department.jmd")`
