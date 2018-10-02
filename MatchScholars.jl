using LinearAlgebra, Statistics
using TextAnalysis, ExcelReaders, StringDistances, AMD, SparseArrays, Plots
cd(@__DIR__)
# This script analyzes the abstracts published by the department of automatic control, Lund University, and looks for similarities between our abstracts and the abstracts of the visiting scholars of the LCCC focus period
# The department abstract and authornames were acuired from https://lup.lub.lu.se/search/publication?q=department+exact+v1000253 and exported as an excel file [publications.xsl](publications.xsl). The abstracts of the visiting scolars are provided in [visitors.txt](visitors.txt)
# To compile this notebook, run the line `Literate.notebook("MatchScholars.jl", ".", documenter=false, execute=false)`

# # Read department data
filename  = "publications.xls"
authors   = readxl(filename, "Sheet1!E2:E1501")[:]
abstracts = readxl(filename, "Sheet1!X2:X1501")[:]

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


# # Read visitor data
visitortext = readlines("visitors.txt")
visitors = map(visitortext) do v
    m = match(r"==name== (.+)$", v)
    if m == nothing
        return m
    end
    String(m.captures[1])
end
visitors = String.(filter!(x->x!=nothing, visitors))
# Assemble the abstract from individual lines
function build_abstract(lines, abstract="", abstracts=String[])
    if length(lines) < 1
        push!(abstracts, abstract)
        return abstracts
    end
    if match(r"==name==", lines[1]) != nothing # Found new name
        push!(abstracts, abstract)
        return build_abstract(lines[2:end], "", abstracts)
    end
    return build_abstract(lines[2:end], abstract*" "*lines[1], abstracts)
end
visitor_abstracts = build_abstract(visitortext)[2:end]

# Prepare data for text analysis
docs = StringDocument.([deepcopy(abstracts); deepcopy(visitor_abstracts)])
crps = Corpus(deepcopy(docs))
prepare!(crps, strip_corrupt_utf8 | strip_case | strip_articles | strip_prepositions | strip_pronouns | strip_stopwords | strip_whitespace | strip_non_letters | strip_numbers)
remove_words!(crps, ["br", "control", "system", "systems"]) # For some reason the word "br" appeas very often (html tag?)
update_lexicon!(crps)

# # Analysis
# We will perform two sets of analysis, [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and [Latent Semantic Analysis (LSA)](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
# Ideally, since we are going to use the founds topics for similarity analysis, we should use a correlated topic model (CTM). I could not find a working implementation of that and didn't have the time to fix one, so LDA will have to do. We can estimate topic correlations adhoc using either ϕ*ϕ' (on a word similarity basis) or θ*θ' (on author similarity basis)

# LDA
m     = DocumentTermMatrix(crps)
k     = 6 # number of topics
iters = 1000 # number of gibbs sampling iterss
α     = 1/k # hyper parameter topics per document
β     = 0.01 # hyber parameter words per topic
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
topic_covariance_by_words     = Matrix(ϕ*ϕ')
topic_covariance_by_documents = Matrix(θ*θ')
topic_correlation_by_words = corr(topic_covariance_by_words)
topic_correlation_by_documents = corr(topic_covariance_by_documents)
document_covariance           = θ'θ

# We now conduct some tedious coding to associate the authors of the abstracts in the publication database with the current staff members
individual_authors = String.(strip.(vcat(split.(authors, ";")...)))
unique_authors     = unique(individual_authors)
author_mapping = map(unique_authors) do ua
    [match(Regex(ua), aa) != nothing for aa in authors]
end
author_mapping = findall.(author_mapping)
author_rank    = sortperm(length.(author_mapping), rev=true)
top_10_authors = unique_authors[author_rank[1:10]]

# The current staff at the department is listed in [staff.txt](staff.txt)
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
staff_authorname = unique_authors[staff2unique_author]

document_indices_of_staff = map(staff_authorname) do staffi
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

# The vectors of the visiting scholars are easy to find since they have only a single abstract each
visitor_vectors = θ[:,end-length(visitors)+1:end] # n_topics × n_visitors

# The correlation between the abstracts of the staff and the abstracts of the visiting scholars are given by the inner product of their respective topic vectors
visitor_staff_covariance = staff_vectors'topic_correlation_by_documents*visitor_vectors

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

C, permutation, ypermutation = diagonalize(visitor_staff_covariance, 4, permute_y=false)
plotcovariance(C, visitors[permutation], staff_authorname[ypermutation], xlabel="Visiting scholars", ylabel="Control staff", size=(600,1000))

# We can do the same analysis among the staff members
staff_covariance = staff_vectors'topic_correlation_by_documents*staff_vectors

C, permutation, ypermutation = diagonalize(staff_covariance, 0.3, permute_y=true, doplot=true)
plotcovariance(C,staff_authorname[permutation],staff_authorname[permutation], xrotation=90, size=(1000,1000), yflip=true)


# LSA
error("LSA not yet working")
tfidf = tf_idf(m)
S = svd(Matrix(tfidf))

staff_vectors = map(document_indices_of_staff) do staffdocinds
    # ϕ: topics × words   θ: topics × documents
    mean(S.U[staffdocinds,1:k], dims=1)[:]
end
staff_vectors = hcat(staff_vectors...) # n_topics × n_staff

# The vectors of the visiting scholars are easy to find since they have only a single abstract each
visitor_vectors = S.U[end-length(visitors)+1:end,1:k]' # n_topics × n_visitors



visitor_staff_covariance = staff_vectors'visitor_vectors

C, permutation, ypermutation = diagonalize(visitor_staff_covariance, 1e-21, permute_y=false, doplot=false)
plotcovariance(C, visitors[permutation], staff_authorname[ypermutation], xlabel="Visiting scholars", ylabel="Control staff", size=(600,1000))

# We can do the same analysis among the staff members
staff_covariance = staff_vectors'topic_correlation_by_documents*staff_vectors

C, permutation, ypermutation = diagonalize(staff_covariance, 0.001, permute_y=true, doplot=true)
plotcovariance(C,staff_authorname[permutation],staff_authorname[permutation], xrotation=90, size=(1000,1000), yflip=true)
