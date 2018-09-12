# chapter 22 - recommender system


import math, random
from collections import defaultdict, Counter
from linear_algebra import dot

users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]



#----- recommendation by popularity
popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests).most_common()
# interest: 관심사 아이템
# user_interests: users_interests 안의 사람별 데이터
# users_interests: 사람별 데이터를 다 모아놓은 하나의 리스트

def most_popular_new_interests(user_interests, max_results=5):
    suggestions = [(interest, frequency)
                   for interest, frequency in popular_interests
                   if interest not in user_interests]
    return suggestions[:max_results]

print(most_popular_new_interests(["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"]))



#----- user-based collaborative filtering

def cosine_similarity(v,w):
    return dot(v,w) / math.sqrt(dot(v,v)*dot(w,w)) # dot 두 행렬의 곱
# 코사인 유사도는 벡터 v,w 사이의 각도를 잰다
# v,w가 같은 방향이면 1, 정반대면 0, v는 있어도 w가 0이면 0이 된다
# 벡터 v는 각 사용자의 관심사
# 유사한 사용자란 벡터끼리 유사한 방향을 가리키는 사용자
unique_interests = sorted(list({ interest
                                 for user_interests in users_interests
                                 for interest in user_interests }))
# 사용자의 관심사 벡터 만들기
# 사용자가 관심사를 가지고 있으면 1, 아니면 0
def make_user_interest_vector(user_interests):
    return [1 if interest in user_interests else 0
            for interest in unique_interests]
user_interest_matrix = map(make_user_interest_vector, users_interests) # map(함수, 인풋)
# user_interest_matrix[i][j]는 사용자 i가 관심사 j에 관심 있을 때 1, 없을 때 0

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                      for interest_vector_j in user_interest_matrix]
                     for interest_vector_i in user_interest_matrix]

def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity)
             for other_user_id, similarity in
             enumerate(user_similarities[user_id])
             if user_id != other_user_id and similarity > 0]

    return sorted(pairs, key=lambda pair: pair[1], reverse=True)
# enumerate는 인덱스 값을 포함하는 enumerate 객체를 리턴한다.
print(most_popular_new_interests(0))

# 각각의 관심사에 대해 해당 관심사에 관심이 있는 다른 사용자와의 유사도 모두 더하기
def user_based_suggestions(user_id, include_current_interests=False):
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
    # 정렬된 리스트로 반환
    suggestions = sorted(suggestions.items(), key=lambda pair: pair[1], reverse=True)
    # 이미 관심사로 표시한 것은 제외
    if include_current_interests:
        return suggestions
    else:
        return [(suggestions, weight)
                for suggestions, weight in suggestions
                if suggestion not in user_interests[user_id]]

user_based_suggestions(0)



#----- item_based_collaborative_filtering

# 관심사가 행, 사용자가 열인 행렬 생성
interest_user_matrix = [[user_interest_vector[j]
                        for user_interest_vector in user_interest_matrix]
                        for j, _ in enumerate(unique_interests)]
interest_user_matrix[0]

# 코사인 유사도 적용
interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                         for user_vector_j in interest_user_matrix]
                         for user_vector_i in interest_user_matrix]

# 특정 관심사와 유사한 관심사 구하기
def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
            for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs, key=lambda pair: pair[1], reverse=True)

print(most_similar_interests_to(0))

